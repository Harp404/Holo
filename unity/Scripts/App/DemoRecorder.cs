using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Holo
{
    /// <summary>
    /// Records the app (frames + Muse's spoken clips with their timestamps) while running a scripted
    /// session: ask -> plan -> guide steps -> web search -> grounded question. Output: ../Recordings/run/
    /// Fixed capture rate, so the video plays at real speed whatever the editor frame rate is.
    /// </summary>
    public class DemoRecorder : MonoBehaviour
    {
        public HoloApp app;
        public PassthroughVideo video;
        public int fps = 24;
        public string outDir;
        int _frame;
        readonly List<string> _speech = new();
        bool _rec;

        readonly List<string> _skips = new();

        /// <summary>Build-mode session. Recording pauses while the GPU works; each skip is logged so the edit can show it.</summary>
        public static void BeginBuild()
        {
            var app = FindFirstObjectByType<HoloApp>();
            var r = app.gameObject.GetComponent<DemoRecorder>() ?? app.gameObject.AddComponent<DemoRecorder>();
            r.app = app;
            r.video = FindFirstObjectByType<PassthroughVideo>();
            r.StartCoroutine(r.RunBuild());
        }

        IEnumerator WaitStage(int stage)
        {
            yield return Sec(1.6f);                       // show the "thinking" state, then skip the wait
            float real = Time.realtimeSinceStartup;
            _rec = false; Time.captureFramerate = 0;
            yield return new WaitUntil(() => app.BuildStage >= stage);
            _skips.Add($"{_frame}\t{Time.realtimeSinceStartup - real + 1.6f:0}");
            Time.captureFramerate = fps; _rec = true;
        }

        IEnumerator WaitIdle()
        {
            yield return Sec(1.6f);
            if (!app.Busy) yield break;
            float real = Time.realtimeSinceStartup;
            _rec = false; Time.captureFramerate = 0;
            yield return new WaitUntil(() => !app.Busy);
            _skips.Add($"{_frame}\t{Time.realtimeSinceStartup - real + 1.6f:0}");
            Time.captureFramerate = fps; _rec = true;
        }

        IEnumerator WaitIdleSeg(float realSecs)
        {   // the highlight request is async with no busy flag: pause recording for a few real seconds
            yield return Sec(0.8f);
            _rec = false; Time.captureFramerate = 0;
            float t = Time.realtimeSinceStartup;
            while (Time.realtimeSinceStartup - t < realSecs) yield return null;
            _skips.Add($"{_frame}\t{realSecs + 0.8f:0}");
            Time.captureFramerate = fps; _rec = true;
            yield return Sec(2.5f);
        }

        IEnumerator Press(string button, float hold = 0.25f)
        {
            var go = GameObject.Find(button);
            var b = go ? go.GetComponent<UnityEngine.UI.Button>() : null;
            if (b == null) { Debug.LogWarning("no button " + button); yield break; }
            var ev = new UnityEngine.EventSystems.PointerEventData(UnityEngine.EventSystems.EventSystem.current);
            UnityEngine.EventSystems.ExecuteEvents.Execute(go, ev, UnityEngine.EventSystems.ExecuteEvents.pointerEnterHandler);
            yield return Sec(0.35f);
            UnityEngine.EventSystems.ExecuteEvents.Execute(go, ev, UnityEngine.EventSystems.ExecuteEvents.pointerDownHandler);
            yield return Sec(hold);
            UnityEngine.EventSystems.ExecuteEvents.Execute(go, ev, UnityEngine.EventSystems.ExecuteEvents.pointerUpHandler);
            b.onClick.Invoke();
            UnityEngine.EventSystems.ExecuteEvents.Execute(go, ev, UnityEngine.EventSystems.ExecuteEvents.pointerExitHandler);
        }

        IEnumerator RunBuild()
        {
            outDir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Recordings", "build"));
            if (Directory.Exists(outDir)) Directory.Delete(outDir, true);
            Directory.CreateDirectory(outDir);
            HoloApp.OnSpeak = clip =>
            {
                var name = $"speech_{_frame:00000}.wav";
                File.WriteAllBytes(Path.Combine(outDir, name), Wav(clip));
                _speech.Add($"{_frame / (float)fps:0.000}\t{name}");
            };
            HoloApp.Recording = true;
            Time.captureFramerate = fps;
            _rec = true;
            video.Seek(1.5); video.Play();
            yield return Sec(1.2f);
            video.Pause();
            yield return app.Say("Muse, I've got these parts. Design me a drone I can build.", 3.0f);
            yield return WaitStage(1);                     // Muse's design
            yield return Sec(1.0f);
            yield return new WaitWhile(() => app.Speaking);
            yield return Sec(0.8f);
            yield return WaitStage(2);                     // render
            yield return Sec(3.0f);
            yield return WaitStage(3);                     // 3D model
            yield return Sec(0.6f);
            yield return new WaitWhile(() => app.Speaking);
            yield return Sec(0.8f);
            yield return Press("View3D");
            yield return Sec(3.0f);
            var v = FindFirstObjectByType<ModelViewer>();
            // drag-rotate (what a pinch-and-turn does on Quest)
            for (float t = 0; t < 2.4f; t += Time.deltaTime) { v.transform.root.Rotate(0, 0, 0); RotateModel(v, 60f * Time.deltaTime); yield return null; }
            yield return Sec(0.6f);
            yield return Press("Top");
            yield return Sec(2.6f);
            yield return Press("Reset");
            yield return Sec(1.6f);
            yield return Press("Spin");
            yield return Sec(3.5f);
            yield return Press("Spin");
            for (float t = 0; t < 1.5f; t += Time.deltaTime) { ScaleModel(v, 1 + 0.45f * Time.deltaTime); yield return null; }
            yield return Sec(1.5f);
            yield return Press("BarSTL");
            yield return Sec(2.0f);
            yield return Press("Close");
            // guide on the real build: Muse plans from the camera frame, SAM 3.1 highlights the step's part
            video.Seek(100 / 12.0); video.Play();
            yield return Sec(1.5f);
            yield return app.Say("Muse, how do I mount this ESC and wire the motors?", 2.8f);
            yield return WaitIdle();
            yield return Sec(1.0f);
            yield return new WaitWhile(() => app.Speaking);
            yield return Sec(1.0f);
            app.StartGuide();
            yield return WaitIdleSeg(4.0f);
            yield return new WaitWhile(() => app.Speaking);
            yield return Sec(1.5f);
            video.Play();
            yield return Sec(2.5f);

            _rec = false;
            Time.captureFramerate = 0;
            File.WriteAllLines(Path.Combine(outDir, "speech.tsv"), _speech);
            File.WriteAllLines(Path.Combine(outDir, "skips.tsv"), _skips);
            HoloApp.Recording = false;
            File.WriteAllText(Path.Combine(outDir, "DONE"), _frame.ToString());
            Debug.Log("[DemoRecorder] build done " + _frame + " frames -> " + outDir);
        }

        static void RotateModel(ModelViewer v, float deg) { var m = GameObject.Find("BuildModel"); if (m) m.transform.Rotate(Vector3.up, deg, Space.World); }
        static void ScaleModel(ModelViewer v, float k) { var m = GameObject.Find("BuildModel"); if (m) m.transform.localScale *= k; }

        public static void Begin()
        {
            var app = FindFirstObjectByType<HoloApp>();
            var r = app.gameObject.GetComponent<DemoRecorder>() ?? app.gameObject.AddComponent<DemoRecorder>();
            r.app = app;
            r.video = FindFirstObjectByType<PassthroughVideo>();
            r.StartCoroutine(r.Run());
        }

        IEnumerator Run()
        {
            outDir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Recordings", "run"));
            if (Directory.Exists(outDir)) Directory.Delete(outDir, true);
            Directory.CreateDirectory(outDir);
            HoloApp.OnSpeak = clip =>
            {
                var name = $"speech_{_frame:00000}.wav";
                File.WriteAllBytes(Path.Combine(outDir, name), Wav(clip));
                _speech.Add($"{_frame / (float)fps:0.000}\t{name}");
            };
            Time.captureFramerate = fps;
            _rec = true;

            // 1) Watching: perception running
            video.Seek(300 / 12.0); video.Play();
            yield return Sec(3.2f);
            // 2) "Muse, how do I connect this cable to my laptop?"
            video.Pause();
            yield return app.Say("Muse, how do I connect this cable to my laptop?");
            yield return new WaitUntil(() => !app.Busy);
            yield return Sec(1.0f);
            yield return new WaitWhile(() => app.Speaking);
            yield return Sec(1.5f);
            // 3) Start the guide, walk 2 steps
            app.StartGuide();
            yield return Sec(1.0f);
            yield return new WaitWhile(() => app.Speaking);
            yield return Sec(1.5f);
            video.Play();
            yield return Sec(2.5f);
            // 4) Mid-task web search at the port
            video.Seek(560 / 12.0); video.Pause();
            yield return Sec(0.6f);
            yield return app.Say("Muse, is this port USB 3.0? Will it charge my phone?", 3.0f);
            yield return new WaitUntil(() => !app.Busy);
            yield return Sec(1.0f);
            yield return new WaitWhile(() => app.Speaking);
            yield return Sec(2.5f);
            // 5) Grounded question about the plug
            video.Seek(600 / 12.0);
            yield return app.Say("Muse, which way up does the plug go in?", 2.4f);
            yield return new WaitUntil(() => !app.Busy);
            yield return Sec(1.0f);
            yield return new WaitWhile(() => app.Speaking);
            yield return Sec(2.0f);
            video.Play();
            yield return Sec(4f);

            _rec = false;
            Time.captureFramerate = 0;
            File.WriteAllLines(Path.Combine(outDir, "speech.tsv"), _speech);
            File.WriteAllText(Path.Combine(outDir, "DONE"), _frame.ToString());
            Debug.Log("[DemoRecorder] done " + _frame + " frames -> " + outDir);
        }

        // game-time wait (captureFramerate makes this deterministic)
        static IEnumerator Sec(float s) { float t = Time.time + s; while (Time.time < t) yield return null; }

        void LateUpdate()
        {
            if (_rec) StartCoroutine(Grab());
        }

        IEnumerator Grab()
        {
            yield return new WaitForEndOfFrame();
            var tex = ScreenCapture.CaptureScreenshotAsTexture();
            File.WriteAllBytes(Path.Combine(outDir, $"{_frame:00000}.jpg"), tex.EncodeToJPG(90));
            Destroy(tex);
            _frame++;
        }

        static byte[] Wav(AudioClip c)
        {
            var data = new float[c.samples * c.channels];
            c.GetData(data, 0);
            using var ms = new MemoryStream();
            using var w = new BinaryWriter(ms);
            int bytes = data.Length * 2;
            w.Write("RIFF".ToCharArray()); w.Write(36 + bytes); w.Write("WAVE".ToCharArray());
            w.Write("fmt ".ToCharArray()); w.Write(16); w.Write((short)1); w.Write((short)c.channels);
            w.Write(c.frequency); w.Write(c.frequency * c.channels * 2); w.Write((short)(c.channels * 2)); w.Write((short)16);
            w.Write("data".ToCharArray()); w.Write(bytes);
            foreach (var s in data) w.Write((short)(Mathf.Clamp(s, -1f, 1f) * short.MaxValue));
            return ms.ToArray();
        }
    }
}
