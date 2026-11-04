using System;
using System.IO;
using UnityEngine;

namespace Holo
{
    /// <summary>
    /// Always-on microphone with the wake word "Muse".
    ///  - Every 0.8 s, if there is speech energy, the last 2.4 s go to /api/wake (faster-whisper tiny).
    ///  - On "Muse", it keeps recording until ~0.9 s of silence (max 10 s), then sends the whole utterance
    ///    (from the window start, so "Muse, how do I..." in one breath works) to /api/stt.
    ///  - Push-to-talk (Hold V / mic button) skips the wake word.
    /// </summary>
    public class WakeListener : MonoBehaviour
    {
        public HoloApi api;
        public int sampleRate = 16000;
        public float windowSeconds = 2.4f;
        public float checkEvery = 0.8f;
        public float speechRms = 0.012f;
        public float silenceSeconds = 0.9f;
        public float maxUtterance = 10f;
        public bool listening = true;

        public enum State { Off, Idle, Waking, Capturing, Transcribing }
        public State Current { get; private set; } = State.Off;
        public float Level { get; private set; }          // 0..1 mic level for the orb / waveform
        public event Action OnWake;
        public event Action<string> OnUtterance;
        public event Action<string> OnStatus;

        AudioClip _clip;
        string _dev;
        const int BufferSeconds = 20;
        float _nextCheck;
        int _capStart;
        float _silentFor, _capFor;
        bool _waitingWake, _pushToTalk;

        void Start()
        {
            if (Microphone.devices.Length == 0) { OnStatus?.Invoke("No microphone found: type instead"); return; }
            _dev = Microphone.devices[0];
            _clip = Microphone.Start(_dev, true, BufferSeconds, sampleRate);
            Current = State.Idle;
            OnStatus?.Invoke("Say “Muse” to talk");
        }

        void OnDisable() { if (_dev != null) Microphone.End(_dev); }

        int Pos => Microphone.GetPosition(_dev);

        float[] Read(int start, int count)
        {
            var data = new float[count];
            int total = _clip.samples;
            start = ((start % total) + total) % total;
            if (start + count <= total) _clip.GetData(data, start);
            else
            {
                var a = new float[total - start];
                var b = new float[count - a.Length];
                _clip.GetData(a, start);
                _clip.GetData(b, 0);
                a.CopyTo(data, 0);
                b.CopyTo(data, a.Length);
            }
            return data;
        }

        static float Rms(float[] x, int from = 0)
        {
            double s = 0;
            for (int i = from; i < x.Length; i++) s += x[i] * x[i];
            return (float)Math.Sqrt(s / Math.Max(1, x.Length - from));
        }

        void Update()
        {
            if (_clip == null) return;
            var kb = UnityEngine.InputSystem.Keyboard.current;
            if (kb != null && kb.vKey.wasPressedThisFrame) BeginPushToTalk();
            if (kb != null && kb.vKey.wasReleasedThisFrame && _pushToTalk) EndCapture();

            // live level (last 0.1 s)
            var recent = Read(Pos - sampleRate / 10, sampleRate / 10);
            Level = Mathf.Lerp(Level, Mathf.Clamp01(Rms(recent) * 18f), 0.4f);

            if (Current == State.Idle && listening && !_waitingWake && Time.time >= _nextCheck)
            {
                _nextCheck = Time.time + checkEvery;
                int n = (int)(windowSeconds * sampleRate);
                var win = Read(Pos - n, n);
                if (Rms(win, n / 2) > speechRms)   // only when someone is talking right now
                {
                    _waitingWake = true;
                    int winStart = Pos - n;
                    StartCoroutine(api.Wake(Wav(win), r =>
                    {
                        _waitingWake = false;
                        if (r.Value<bool>("wake") && Current == State.Idle) BeginCapture(winStart);
                    }, e => { _waitingWake = false; Debug.LogWarning(e); }));
                }
            }

            if (Current == State.Capturing)
            {
                _capFor += Time.deltaTime;
                if (!_pushToTalk)
                {
                    _silentFor = Rms(recent) < speechRms * 0.8f ? _silentFor + Time.deltaTime : 0f;
                    if ((_silentFor > silenceSeconds && _capFor > 1.2f) || _capFor > maxUtterance) EndCapture();
                }
                else if (_capFor > maxUtterance) EndCapture();
            }
        }

        public void BeginPushToTalk()
        {
            if (_clip == null || Current == State.Capturing) return;
            _pushToTalk = true;
            BeginCapture(Pos);
        }

        void BeginCapture(int fromSample)
        {
            _capStart = fromSample;
            _silentFor = 0;
            _capFor = 0;
            Current = State.Capturing;
            OnWake?.Invoke();
            OnStatus?.Invoke("Listening…");
        }

        public void EndCapture()
        {
            if (Current != State.Capturing) return;
            int end = Pos;
            int total = _clip.samples;
            int len = ((end - _capStart) % total + total) % total;
            _pushToTalk = false;
            if (len < sampleRate / 3) { Current = State.Idle; OnStatus?.Invoke("Say “Muse” to talk"); return; }
            var data = Read(_capStart, len);
            Current = State.Transcribing;
            OnStatus?.Invoke("Transcribing…");
            StartCoroutine(api.Stt(Wav(data), r =>
            {
                Current = State.Idle;
                _nextCheck = Time.time + 1.5f;
                var t = ((string)r["text"] ?? "").Trim();
                OnStatus?.Invoke("Say “Muse” to talk");
                if (t.Length > 1) OnUtterance?.Invoke(t);
            }, e => { Current = State.Idle; Debug.LogWarning(e); OnStatus?.Invoke("Voice error: type instead"); }));
        }

        string Wav(float[] samples)
        {
            using var ms = new MemoryStream();
            using var w = new BinaryWriter(ms);
            int bytes = samples.Length * 2;
            w.Write("RIFF".ToCharArray()); w.Write(36 + bytes); w.Write("WAVE".ToCharArray());
            w.Write("fmt ".ToCharArray()); w.Write(16); w.Write((short)1); w.Write((short)1);
            w.Write(sampleRate); w.Write(sampleRate * 2); w.Write((short)2); w.Write((short)16);
            w.Write("data".ToCharArray()); w.Write(bytes);
            foreach (var s in samples) w.Write((short)(Mathf.Clamp(s, -1f, 1f) * short.MaxValue));
            return Convert.ToBase64String(ms.ToArray());
        }
    }
}
