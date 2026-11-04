using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands;
using UnityEngine.XR.Management;

namespace Holo
{
    /// <summary>
    /// Shows a generated 3D model as a hologram in front of the user and lets them handle it.
    ///   Quest (XR Hands):  pinch near the model to grab -> move + rotate with your hand;
    ///                      pinch with both hands -> scale and turn (like Meta's two-hand transformer)
    ///   Head:              Follow mode keeps it lazily in front of you; Pinned leaves it in space so you
    ///                      can walk around it / look from the top
    ///   Editor fallback:   left-drag rotate, right-drag move, scroll scale
    ///   Views:             top view, turntable spin, reset
    /// </summary>
    public class ModelViewer : MonoBehaviour
    {
        public Camera head;
        public float distance = 0.62f;
        public float drop = 0.05f;          // below eye level so you look down onto it (3/4 view)
        public float targetSize = 0.36f;
        public bool follow = true;
        public bool spin;
        public event Action<string> OnStatus;

        Transform _root, _model;
        Material _holo;
        XRHandSubsystem _hands;
        bool _grabL, _grabR;
        Vector3 _grabOffset;
        Quaternion _grabRot;
        float _twoDist, _twoScale;
        Vector3 _twoDir;
        Quaternion _twoRot;
        Quaternion _targetRot = Quaternion.identity;
        bool _animRot;
        float _appear;

        public bool Visible => _root != null && _root.gameObject.activeSelf;

        void Awake()
        {
            var sh = Shader.Find("Holo/HoloModel");
            if (sh != null) _holo = new Material(sh);
        }

        public IEnumerator Show(byte[] glb)
        {
            Clear();
            var go = new GameObject("BuildModel");
            _root = go.transform;
            var holder = new GameObject("Model").transform;
            holder.SetParent(_root, false);
            var gltf = new GLTFast.GltfImport();
            var load = gltf.LoadGltfBinary(glb);
            while (!load.IsCompleted) yield return null;
            if (!load.Result) { OnStatus?.Invoke("Couldn't read the 3D model"); yield break; }
            var inst = gltf.InstantiateMainSceneAsync(holder);
            while (!inst.IsCompleted) yield return null;
            _model = holder;
            // hologram material everywhere, centre + normalise size
            var rends = holder.GetComponentsInChildren<Renderer>();
            var b = new Bounds();
            bool first = true;
            foreach (var r in rends)
            {
                if (_holo != null) { var mats = new Material[r.sharedMaterials.Length]; for (int i = 0; i < mats.Length; i++) mats[i] = _holo; r.sharedMaterials = mats; }
                if (first) { b = r.bounds; first = false; } else b.Encapsulate(r.bounds);
            }
            float s = targetSize / Mathf.Max(1e-4f, Mathf.Max(b.size.x, Mathf.Max(b.size.y, b.size.z)));
            holder.localScale = Vector3.one * s;
            holder.localPosition = -b.center * s;
            _root.position = head.transform.position + FlatForward() * distance + Vector3.down * drop;
            _root.rotation = Present();
            Pedestal(b.size.y * s);
            _targetRot = _root.rotation;
            _appear = 0;
            OnStatus?.Invoke("Pinch to grab · two hands to scale · move your head to look around");
        }

        Transform _ped;

        /// <summary>Glowing ring under the model (VR model-viewer pedestal): grounds it in space, shows its footprint.</summary>
        void Pedestal(float height)
        {
            var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
            Destroy(q.GetComponent<Collider>());
            q.name = "Pedestal";
            _ped = q.transform;
            _ped.SetParent(_root, false);
            _ped.localPosition = new Vector3(0, -height * 0.5f - 0.035f, 0);
            _ped.localRotation = Quaternion.Euler(90, 0, 0);
            _ped.localScale = Vector3.one * targetSize * 1.25f;
            const int N = 256;
            var tex = new Texture2D(N, N, TextureFormat.RGBA32, false);
            var px = new Color32[N * N];
            for (int y = 0; y < N; y++)
                for (int x = 0; x < N; x++)
                {
                    float r = Vector2.Distance(new Vector2(x, y), new Vector2(N / 2f, N / 2f)) / (N / 2f);
                    float ring = 0.55f * Mathf.Exp(-Mathf.Pow((r - 0.86f) / 0.02f, 2)) + 0.2f * Mathf.Exp(-Mathf.Pow((r - 0.86f) / 0.09f, 2));
                    float fill = r < 0.86f ? 0.10f * (1 - r) : 0;
                    float a = Mathf.Clamp01(ring + fill);
                    px[y * N + x] = new Color(0.55f, 0.8f, 1f, a);
                }
            tex.SetPixels32(px); tex.Apply();
            var m = new Material(Shader.Find("Sprites/Default")) { mainTexture = tex };
            q.GetComponent<Renderer>().material = m;
        }

        public void Clear()
        {
            if (_root != null) Destroy(_root.gameObject);
            _root = null;
            _model = null;
        }

        public void ToggleFollow() { follow = !follow; OnStatus?.Invoke(follow ? "Following your view" : "Pinned in place: walk around it"); }
        public void ToggleSpin() => spin = !spin;
        public void TopView() { if (_root == null) return; _targetRot = Quaternion.LookRotation(FlatForward(), Vector3.up) * Quaternion.Euler(80, 0, 0); _animRot = true; }
        public void ResetView() { if (_root == null) return; _targetRot = Present(); _animRot = true; _root.localScale = Vector3.one; }

        /// <summary>Default pose: facing you 3/4, tipped 22 deg toward your eyes so you see the top too.</summary>
        Quaternion Present() => Quaternion.AngleAxis(22, Vector3.Cross(Vector3.up, FlatForward())) * Quaternion.LookRotation(FlatForward()) * Quaternion.Euler(0, 200, 0);

        Vector3 FlatForward()
        {
            var f = head.transform.forward; f.y = 0;
            return f.sqrMagnitude < 1e-4f ? Vector3.forward : f.normalized;
        }

        bool Pinch(XRHand h, out Vector3 p)
        {
            p = default;
            if (!h.isTracked) return false;
            if (!h.GetJoint(XRHandJointID.ThumbTip).TryGetPose(out var a) || !h.GetJoint(XRHandJointID.IndexTip).TryGetPose(out var b)) return false;
            var o = head.transform.parent != null ? head.transform.parent : null;   // XR origin space -> world
            Vector3 pa = o ? o.TransformPoint(a.position) : a.position, pb = o ? o.TransformPoint(b.position) : b.position;
            p = (pa + pb) * 0.5f;
            return Vector3.Distance(pa, pb) < 0.022f;
        }

        void Update()
        {
            if (_root == null) return;
            _appear = Mathf.MoveTowards(_appear, 1, Time.deltaTime * 2.5f);
            if (_holo != null) _holo.SetFloat("_Alpha", _appear);
            float dt = Time.deltaTime;

            // ---- hands (Quest) ----
            if (_hands == null) _hands = XRGeneralSettings.Instance?.Manager?.activeLoader?.GetLoadedSubsystem<XRHandSubsystem>();
            bool l = false, r = false;
            Vector3 pl = default, pr = default;
            if (_hands != null)
            {
                l = Pinch(_hands.leftHand, out pl);
                r = Pinch(_hands.rightHand, out pr);
            }
            float reach = targetSize * _root.localScale.x * 0.9f + 0.08f;
            if (l && r && (_grabL || _grabR || Vector3.Distance((pl + pr) / 2, _root.position) < reach * 1.5f))
            {
                var dir = pr - pl;
                if (!(_grabL && _grabR)) { _twoDist = dir.magnitude; _twoScale = _root.localScale.x; _twoDir = dir; _twoRot = _root.rotation; }
                _grabL = _grabR = true;
                float k = Mathf.Clamp(dir.magnitude / Mathf.Max(0.01f, _twoDist), 0.25f, 4f);
                _root.localScale = Vector3.one * Mathf.Clamp(_twoScale * k, 0.3f, 6f);
                var a = new Vector3(_twoDir.x, 0, _twoDir.z); var c = new Vector3(dir.x, 0, dir.z);
                _root.rotation = Quaternion.FromToRotation(a, c) * _twoRot;
                _root.position = Vector3.Lerp(_root.position, (pl + pr) / 2, 0.35f);
                follow = false; _animRot = false;
            }
            else if (r || l)
            {
                var p = r ? pr : pl;
                bool was = r ? _grabR : _grabL;
                if (!was && Vector3.Distance(p, _root.position) < reach) { _grabOffset = _root.position - p; _grabRot = _root.rotation; }
                if (was || Vector3.Distance(p, _root.position) < reach)
                {
                    if (r) _grabR = true; else _grabL = true;
                    _root.position = Vector3.Lerp(_root.position, p + _grabOffset, 0.5f);
                    follow = false; _animRot = false;
                }
            }
            else { _grabL = _grabR = false; }

            // ---- editor / mouse fallback ----
            var mouse = UnityEngine.InputSystem.Mouse.current;
            bool overUi = UnityEngine.EventSystems.EventSystem.current != null && UnityEngine.EventSystems.EventSystem.current.IsPointerOverGameObject();
            if (mouse != null && !overUi)
            {
                var d = mouse.delta.ReadValue();
                if (mouse.leftButton.isPressed) { _root.Rotate(head.transform.up, -d.x * 0.35f, Space.World); _root.Rotate(head.transform.right, d.y * 0.35f, Space.World); _animRot = false; spin = false; }
                if (mouse.rightButton.isPressed) { _root.position += (head.transform.right * d.x + head.transform.up * d.y) * 0.0015f; follow = false; }
                float sc = mouse.scroll.ReadValue().y;
                if (Mathf.Abs(sc) > 0.01f) _root.localScale = Vector3.one * Mathf.Clamp(_root.localScale.x * (1 + Mathf.Sign(sc) * 0.08f), 0.3f, 6f);
            }

            // ---- head follow (lazy) / views ----
            if (follow && !_grabL && !_grabR)
            {
                var want = head.transform.position + FlatForward() * distance + Vector3.down * drop;
                if (Vector3.Distance(want, _root.position) > 0.12f) _root.position = Vector3.Lerp(_root.position, want, dt * 2.5f);
            }
            if (spin && !_grabL && !_grabR) _root.Rotate(Vector3.up, 25f * dt, Space.World);
            if (_animRot) { _root.rotation = Quaternion.Slerp(_root.rotation, _targetRot, dt * 4f); if (Quaternion.Angle(_root.rotation, _targetRot) < 0.5f) _animRot = false; }
        }
    }
}
