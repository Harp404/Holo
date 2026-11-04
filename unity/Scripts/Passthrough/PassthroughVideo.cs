using System;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

namespace Holo
{
    /// <summary>
    /// Stand-in for the headset passthrough camera: plays a real first-person recording (JPEG frame
    /// sequence in StreamingAssets) full-screen and hands the current frame to the model.
    /// On Quest this is replaced by the Passthrough Camera API feed.
    /// </summary>
    public class PassthroughVideo : MonoBehaviour
    {
        public RawImage display;
        [Tooltip("Folder under StreamingAssets with 0001.jpg, 0002.jpg, ...")]
        public string framesFolder = "Passthrough/pov";
        [Tooltip("Optional: raw camera frames (same numbering) sent to the model instead of the displayed ones")]
        public string rawFolder = "";
        public float fps = 12f;
        public bool loop = true;
        public int grabWidth = 1024;
        [Range(30, 95)] public int jpegQuality = 82;

        string[] _files;
        Texture2D _frame;
        RenderTexture _small;
        Texture2D _grab;
        int _shown = -1;
        double _t;
        bool _playing;

        public bool IsPlaying => _playing;
        public double Time => _t;
        public double Duration => _files == null ? 0 : _files.Length / fps;

        void Awake()
        {
            var dir = Path.Combine(Application.streamingAssetsPath, framesFolder);
            _files = Directory.Exists(dir) ? Directory.GetFiles(dir, "*.jpg") : Array.Empty<string>();
            Array.Sort(_files, StringComparer.Ordinal);
            _frame = new Texture2D(2, 2, TextureFormat.RGB24, false);
            if (display != null) display.texture = _frame;
            ShowFrame(0);
        }

        void Update()
        {
            if (!_playing || _files.Length == 0) return;
            _t += UnityEngine.Time.deltaTime;
            if (_t >= Duration) { if (loop) _t %= Duration; else { _t = Duration - 1e-3; _playing = false; } }
            ShowFrame((int)(_t * fps));
            if (display != null && display.texture != _frame) display.texture = _frame;
        }

        void ShowFrame(int i)
        {
            if (_files.Length == 0) return;
            i = Mathf.Clamp(i, 0, _files.Length - 1);
            if (i == _shown) return;
            _frame.LoadImage(File.ReadAllBytes(_files[i]));
            _shown = i;
            if (display != null && display.texture != _frame) display.texture = _frame;
        }

        public void Play() => _playing = true;
        public void Pause() => _playing = false;
        public void Toggle() => _playing = !_playing;
        public void ShowFirstFrame() { _t = 0; _playing = false; ShowFrame(0); }
        public void Seek(double seconds) { _t = Math.Clamp(seconds, 0, Math.Max(0, Duration - 1e-3)); ShowFrame((int)(_t * fps)); }

        public int FrameIndex => Mathf.Max(0, _shown);

        public string GrabBase64()
        {
            if (!string.IsNullOrEmpty(rawFolder) && _shown >= 0)
            {
                var raw = Path.Combine(Application.streamingAssetsPath, rawFolder, Path.GetFileName(_files[_shown]));
                if (File.Exists(raw)) return Convert.ToBase64String(File.ReadAllBytes(raw));
            }
            int h = Mathf.RoundToInt(grabWidth * (float)_frame.height / Mathf.Max(1, _frame.width));
            if (_small == null || _small.height != h)
            {
                _small = new RenderTexture(grabWidth, h, 0);
                _grab = new Texture2D(grabWidth, h, TextureFormat.RGB24, false);
            }
            Graphics.Blit(_frame, _small);
            var prev = RenderTexture.active;
            RenderTexture.active = _small;
            _grab.ReadPixels(new Rect(0, 0, grabWidth, h), 0, 0);
            _grab.Apply();
            RenderTexture.active = prev;
            return Convert.ToBase64String(_grab.EncodeToJPG(jpegQuality));
        }

        /// <summary>Rect (in the display's local space) for a 0-1000 image box.</summary>
        public Rect BoxToRect(float[] box)
        {
            var r = display.rectTransform.rect;
            float x1 = box[0] / 1000f, y1 = box[1] / 1000f, x2 = box[2] / 1000f, y2 = box[3] / 1000f;
            return Rect.MinMaxRect(r.xMin + x1 * r.width, r.yMax - y2 * r.height, r.xMin + x2 * r.width, r.yMax - y1 * r.height);
        }
    }
}
