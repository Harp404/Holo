using System;
using System.IO;
using UnityEngine;

namespace Holo
{
    /// <summary>Push-to-talk: hold V (desktop) to record; the clip is transcribed by the server.</summary>
    public class VoiceInput : MonoBehaviour
    {
        public HoloApi api;
        public int sampleRate = 16000;
        public int maxSeconds = 15;
        public event Action<string> OnText;
        public bool Recording { get; private set; }

        AudioClip _clip;
        string _device;

        void Update()
        {
            var kb = UnityEngine.InputSystem.Keyboard.current;
            if (kb == null) return;
            if (kb.vKey.wasPressedThisFrame) Begin();
            if (kb.vKey.wasReleasedThisFrame) End();
        }

        public void Begin()
        {
            if (Recording || Microphone.devices.Length == 0) return;
            _device = Microphone.devices[0];
            _clip = Microphone.Start(_device, false, maxSeconds, sampleRate);
            Recording = true;
        }

        public void End()
        {
            if (!Recording) return;
            int samples = Microphone.GetPosition(_device);
            Microphone.End(_device);
            Recording = false;
            if (samples < sampleRate / 4) return; // under 0.25 s, ignore

            var data = new float[samples * _clip.channels];
            _clip.GetData(data, 0);
            var wav = Convert.ToBase64String(ToWav(data, _clip.channels, sampleRate));
            StartCoroutine(api.Transcribe(wav,
                r => { var t = (string)r["text"]; if (!string.IsNullOrWhiteSpace(t)) OnText?.Invoke(t.Trim()); },
                e => Debug.LogWarning(e)));
        }

        static byte[] ToWav(float[] samples, int channels, int rate)
        {
            using var ms = new MemoryStream();
            using var w = new BinaryWriter(ms);
            int bytes = samples.Length * 2;
            w.Write("RIFF".ToCharArray()); w.Write(36 + bytes); w.Write("WAVE".ToCharArray());
            w.Write("fmt ".ToCharArray()); w.Write(16); w.Write((short)1); w.Write((short)channels);
            w.Write(rate); w.Write(rate * channels * 2); w.Write((short)(channels * 2)); w.Write((short)16);
            w.Write("data".ToCharArray()); w.Write(bytes);
            foreach (var s in samples) w.Write((short)(Mathf.Clamp(s, -1f, 1f) * short.MaxValue));
            return ms.ToArray();
        }
    }
}
