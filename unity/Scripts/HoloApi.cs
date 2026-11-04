using System;
using System.Collections;
using System.Text;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;

namespace Holo
{
    /// <summary>
    /// Talks to the local Holo server (holo/server.py), which forwards to Muse Spark Open.
    /// Keeps API keys out of the Unity build. All boxes are [x1,y1,x2,y2] in 0-1000 image space.
    /// </summary>
    public class HoloApi : MonoBehaviour
    {
        public string serverUrl = "http://127.0.0.1:8765";

        public IEnumerator Plan(string imageB64, string goal, string mode, string[] objects, Action<JObject> done, Action<string> fail)
        {
            var body = new JObject { ["image"] = imageB64, ["goal"] = goal, ["mode"] = mode };
            if (objects != null) body["objects"] = new JArray(objects);
            yield return Post("/api/plan", body, done, fail);
        }

        public IEnumerator Check(string imageB64, string goal, JObject step, string mode, Action<JObject> done, Action<string> fail)
        {
            var body = new JObject { ["image"] = imageB64, ["goal"] = goal, ["step"] = step, ["mode"] = mode };
            yield return Post("/api/check", body, done, fail);
        }

        public IEnumerator Ask(string imageB64, string question, string goal, JObject step, Action<JObject> done, Action<string> fail)
        {
            var body = new JObject { ["image"] = imageB64, ["question"] = question, ["goal"] = goal, ["step"] = step };
            yield return Post("/api/ask", body, done, fail);
        }

        public IEnumerator Search(string imageB64, string question, string goal, Action<JObject> done, Action<string> fail)
        {
            var body = new JObject { ["image"] = imageB64, ["question"] = question, ["goal"] = goal };
            yield return Post("/api/search", body, done, fail);
        }

        public IEnumerator Transcribe(string wavB64, Action<JObject> done, Action<string> fail)
        {
            var body = new JObject { ["audio"] = wavB64 };
            yield return Post("/api/transcribe", body, done, fail);
        }

        /// <summary>One entry point: Muse routes to a plan, a grounded answer, or an agentic web search.</summary>
        public IEnumerator Chat(string imageB64, string text, string goal, Action<JObject> done, Action<string> fail)
        {
            var body = new JObject { ["image"] = imageB64, ["text"] = text, ["goal"] = goal };
            yield return Post("/api/chat", body, done, fail);
        }

        /// <summary>SAM 3.1 highlight of exactly the named thing: returns a glow overlay PNG in the role colour.</summary>
        public IEnumerator Segment(string imageB64, string text, string role, Action<JObject> done, Action<string> fail)
        {
            var body = new JObject { ["image"] = imageB64, ["text"] = text, ["role"] = role };
            yield return Post("/api/segment", body, done, fail);
        }

        /// <summary>Build mode: Muse designs it, then image + 3D model (GLB) + STL for printing. Slow (~1-2 min).</summary>
        public IEnumerator Build(string imageB64, string goal, Action<JObject> done, Action<string> fail)
        {
            var body = new JObject { ["image"] = imageB64, ["goal"] = goal };
            yield return Post("/api/build", body, done, fail, 900);
        }

        // staged build: the card fills in as each stage lands
        public IEnumerator BuildDesign(string imageB64, string goal, Action<JObject> done, Action<string> fail)
            => Post("/api/build/design", new JObject { ["image"] = imageB64, ["goal"] = goal }, done, fail, 120);
        public IEnumerator BuildRender(string id, Action<JObject> done, Action<string> fail)
            => Post($"/api/build/{id}/render", new JObject(), done, fail, 600);
        public IEnumerator BuildMesh(string id, Action<JObject> done, Action<string> fail)
            => Post($"/api/build/{id}/mesh", new JObject(), done, fail, 900);
        public IEnumerator GetBuild(string id, Action<JObject> done, Action<string> fail)
        {
            using var req = UnityWebRequest.Get(serverUrl + "/api/build/" + id);
            req.timeout = 60;
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success) { fail?.Invoke(req.error); yield break; }
            done?.Invoke(JObject.Parse(req.downloadHandler.text));
        }

        public IEnumerator Wake(string wavB64, Action<JObject> done, Action<string> fail)
        {
            yield return Post("/api/wake", new JObject { ["audio"] = wavB64 }, done, fail);
        }

        public IEnumerator Stt(string wavB64, Action<JObject> done, Action<string> fail)
        {
            yield return Post("/api/stt", new JObject { ["audio"] = wavB64 }, done, fail);
        }

        public IEnumerator Tts(string text, Action<JObject> done, Action<string> fail)
        {
            yield return Post("/api/tts", new JObject { ["text"] = text }, done, fail);
        }

        IEnumerator Post(string path, JObject body, Action<JObject> done, Action<string> fail, int timeout = 120)
        {
            using var req = new UnityWebRequest(serverUrl + path, "POST");
            req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body.ToString()));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = timeout;
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                fail?.Invoke($"{path}: {req.error} {req.downloadHandler?.text}");
                yield break;
            }
            JObject json;
            try { json = JObject.Parse(req.downloadHandler.text); }
            catch (Exception e) { fail?.Invoke($"{path}: bad JSON {e.Message}"); yield break; }
            done?.Invoke(json);
        }
    }
}
