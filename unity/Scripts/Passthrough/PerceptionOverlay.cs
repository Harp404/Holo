using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace Holo
{
    /// <summary>
    /// Renders perception over the passthrough view, synced to the current video frame:
    ///  - SAM 3 masks: the target part glows in its exact shape (fill + animated edge), other parts get a faint edge.
    ///  - hands: Quest-style blue glow on the hand mask plus glowing joints and bones.
    /// </summary>
    public class PerceptionOverlay : MonoBehaviour
    {
        public SceneTrack track;
        public PassthroughVideo feed;
        public RawImage maskImage;       // stretched over the video display, material = Holo/MaskOverlay
        public RectTransform handLayer;  // stretched over the video display
        public Color handColor = new(0.35f, 0.68f, 1f, 1f);
        public Color targetColor = new(0.36f, 0.85f, 1f, 1f);
        public bool showOthers = true;
        public bool showHands = true;

        public int Target { get; set; }
        public int CurrentFrame { get; private set; }

        Texture2D _palette;
        Material _mat;
        readonly List<Image> _bones = new(), _joints = new();
        Sprite _dot;

        static readonly int[,] Bones =
        {
            {0,1},{1,2},{2,3},{3,4}, {0,5},{5,6},{6,7},{7,8}, {5,9},{9,10},{10,11},{11,12},
            {9,13},{13,14},{14,15},{15,16}, {13,17},{0,17},{17,18},{18,19},{19,20}
        };

        void Start()
        {
            _mat = new Material(Shader.Find("Holo/MaskOverlay"));
            maskImage.material = _mat;
            maskImage.raycastTarget = false;
            _palette = new Texture2D(256, 1, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
            _mat.SetTexture("_Palette", _palette);
            _dot = MakeDot();
            RebuildPalette();
            maskImage.enabled = track != null && track.Loaded;
        }

        void RebuildPalette()
        {
            var px = new Color32[256];
            for (int k = 1; k <= (track != null ? track.Names.Count : 0) && k < 256; k++)
            {
                Color c;
                if (track.IsHand(k)) c = new Color(handColor.r, handColor.g, handColor.b, showHands ? 0.32f : 0f);
                else if (k == Target) c = new Color(targetColor.r, targetColor.g, targetColor.b, 1f);
                else c = new Color(1f, 1f, 1f, showOthers ? 0.07f : 0f);
                px[k] = c;
            }
            _palette.SetPixels32(px);
            _palette.Apply();
            _mat.SetFloat("_Target", Target > 0 ? Target : -1);
        }

        int _lastTarget = -2;

        void LateUpdate()
        {
            if (track == null || !track.Loaded || feed == null) return;
            CurrentFrame = track.FrameAt(feed.Time);
            if (Target != _lastTarget) { RebuildPalette(); _lastTarget = Target; }

            var mask = track.Mask(CurrentFrame);
            maskImage.texture = mask;
            if (mask != null) _mat.SetVector("_TexelSize", new Vector4(1f / mask.width, 1f / mask.height, 0, 0));

            DrawHands(track.Frame(CurrentFrame));
        }

        void DrawHands(JObject frame)
        {
            int nb = 0, nj = 0;
            if (showHands && frame != null && frame["hands"] is JArray hands)
            {
                var r = handLayer.rect;
                foreach (JObject h in hands)
                {
                    var kp = (JArray)h["kp"];
                    if (kp == null || kp.Count < 21) continue;
                    var p = new Vector2[21];
                    for (int j = 0; j < 21; j++)
                        p[j] = new Vector2(r.xMin + (float)kp[j][0] * r.width, r.yMax - (float)kp[j][1] * r.height);
                    for (int b = 0; b < Bones.GetLength(0); b++)
                    {
                        var img = Get(_bones, nb++, false);
                        var a = p[Bones[b, 0]];
                        var c = p[Bones[b, 1]];
                        var rt = img.rectTransform;
                        rt.anchoredPosition = (a + c) * 0.5f;
                        rt.sizeDelta = new Vector2((c - a).magnitude, 7f);
                        rt.localRotation = Quaternion.Euler(0, 0, Mathf.Atan2(c.y - a.y, c.x - a.x) * Mathf.Rad2Deg);
                    }
                    for (int j = 0; j < 21; j++)
                    {
                        var img = Get(_joints, nj++, true);
                        img.rectTransform.anchoredPosition = p[j];
                        float s = j % 4 == 0 && j > 0 ? 16f : 11f; // fingertips a bit bigger
                        img.rectTransform.sizeDelta = new Vector2(s, s);
                    }
                }
            }
            for (int i = nb; i < _bones.Count; i++) _bones[i].enabled = false;
            for (int i = nj; i < _joints.Count; i++) _joints[i].enabled = false;
        }

        Image Get(List<Image> pool, int i, bool dot)
        {
            while (pool.Count <= i)
            {
                var go = new GameObject(dot ? "Joint" : "Bone", typeof(RectTransform), typeof(Image));
                go.transform.SetParent(handLayer, false);
                var img = go.GetComponent<Image>();
                img.raycastTarget = false;
                img.sprite = _dot;
                img.type = dot ? Image.Type.Simple : Image.Type.Sliced;
                img.color = dot ? new Color(0.85f, 0.95f, 1f, 0.95f) : new Color(handColor.r, handColor.g, handColor.b, 0.75f);
                var glow = go.AddComponent<Shadow>();
                glow.effectColor = new Color(handColor.r, handColor.g, handColor.b, 0.5f);
                glow.effectDistance = Vector2.zero;
                pool.Add(img);
            }
            pool[i].enabled = true;
            return pool[i];
        }

        static Sprite MakeDot()
        {
            const int n = 32;
            var t = new Texture2D(n, n, TextureFormat.RGBA32, false);
            for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float d = Vector2.Distance(new Vector2(x + 0.5f, y + 0.5f), new Vector2(n / 2f, n / 2f));
                t.SetPixel(x, y, new Color(1, 1, 1, Mathf.Clamp01((n / 2f - d) / 2f)));
            }
            t.Apply();
            return Sprite.Create(t, new Rect(0, 0, n, n), new Vector2(0.5f, 0.5f), 100, 0, SpriteMeshType.FullRect, new Vector4(12, 12, 12, 12));
        }
    }
}
