using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Holo
{
    /// <summary>
    /// Per-frame perception for the passthrough feed, produced by holo/perception/perceive.py
    /// (Muse Spark inventory -> SAM 3 tracking -> DINOv3 identity -> hand joints -> contacts).
    /// Folder under StreamingAssets: scene.json + masks/0001.png ...
    /// </summary>
    public class SceneTrack : MonoBehaviour
    {
        public string folder = "Perception/pov";

        public bool Loaded { get; private set; }
        public float Fps { get; private set; } = 15;
        public int FrameCount => _frames?.Count ?? 0;
        public readonly List<string> Names = new();     // index-1 -> object name
        public readonly List<string> Labels = new();    // index-1 -> SAM 3 concept label

        JArray _frames;
        string _maskDir;
        Texture2D _mask;
        int _maskShown = -1;

        void Awake()
        {
            var dir = Path.Combine(Application.streamingAssetsPath, folder);
            var json = Path.Combine(dir, "scene.json");
            if (!File.Exists(json)) { Debug.Log("[Holo] no perception track at " + dir); return; }
            var scene = JObject.Parse(File.ReadAllText(json));
            Fps = (float?)scene["fps"] ?? 15f;
            foreach (var o in (JArray)scene["objects"])
            {
                Names.Add((string)o["name"]);
                Labels.Add((string)o["label"]);
            }
            _frames = (JArray)scene["frames"];
            _maskDir = Path.Combine(dir, "masks");
            _mask = new Texture2D(2, 2, TextureFormat.RGB24, false) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
            Loaded = true;
            Debug.Log($"[Holo] perception: {FrameCount} frames, objects: {string.Join(", ", Names)}");
        }

        public int FrameAt(double seconds) => Mathf.Clamp((int)(seconds * Fps), 0, Mathf.Max(0, FrameCount - 1));

        public JObject Frame(int i) => Loaded && i >= 0 && i < FrameCount ? (JObject)_frames[i] : null;

        /// <summary>Label map for frame i (red channel = object index).</summary>
        public Texture2D Mask(int i)
        {
            if (!Loaded) return null;
            if (i != _maskShown)
            {
                var p = Path.Combine(_maskDir, $"{i + 1:0000}.png");
                if (File.Exists(p)) _mask.LoadImage(File.ReadAllBytes(p));
                _mask.filterMode = FilterMode.Point;
                _maskShown = i;
            }
            return _mask;
        }

        /// <summary>1-based object index for a name the model used (exact, then fuzzy), or 0.</summary>
        public int Find(string name)
        {
            if (string.IsNullOrEmpty(name)) return 0;
            var n = name.Trim().ToLowerInvariant();
            for (int k = 0; k < Names.Count; k++) if (Names[k].ToLowerInvariant() == n) return k + 1;
            for (int k = 0; k < Names.Count; k++) if (n.Contains(Names[k].ToLowerInvariant()) || Names[k].ToLowerInvariant().Contains(n)) return k + 1;
            for (int k = 0; k < Labels.Count; k++) if (n.Contains(Labels[k].ToLowerInvariant())) return k + 1;
            return 0;
        }

        public bool IsHand(int index) => index > 0 && index <= Labels.Count && Labels[index - 1] == "hand";

        /// <summary>Box (0-1000) of an object in frame i, if visible.</summary>
        public bool TryBox(int frame, int index, out float[] box)
        {
            box = null;
            var f = Frame(frame);
            if (f == null) return false;
            foreach (var o in (JArray)f["objects"])
                if ((int)o["index"] == index)
                {
                    var b = (JArray)o["box"];
                    box = new[] { (float)b[0], (float)b[1], (float)b[2], (float)b[3] };
                    return true;
                }
            return false;
        }

        /// <summary>"grasp" / "touch" / null for the hands and an object in frame i.</summary>
        public string Contact(int frame, int index)
        {
            var f = Frame(frame);
            if (f == null) return null;
            string best = null;
            foreach (var c in (JArray)f["contacts"])
                if ((int)c["object"] == index)
                {
                    var t = (string)c["type"];
                    if (t == "grasp") return t;
                    best = t;
                }
            return best;
        }
    }
}
