using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Holo
{
    /// <summary>
    /// The Holo app (desktop stand-in for Quest passthrough).
    ///   background  first-person view with Holo perception (Meta hands + object covers)
    ///   Muse orb    bottom centre: idle / listening / thinking / speaking. Tap it, press M, or say "Muse".
    ///   panel       floating Meta-style window in front: search/ask bar (type or voice), chat, plans,
    ///               web results with sources, quick actions
    ///   guide       step card + target highlight from Muse's plan; N / B to move between steps
    /// Keys: M open/close · double-tap view · Hold V talk · Enter send · Space play/pause · N/B step · Esc close
    /// </summary>
    public class HoloApp : MonoBehaviour
    {
        [Header("Wiring")]
        public Camera cam;
        public PassthroughVideo video;
        public HoloApi api;
        public WakeListener wake;
        public ModelViewer viewer;
        public AudioSource voice;
        public Font font;
        public Texture2D icMic, icSend, icSearch, icClose, icMenu, icCheck, icArrow, icBulb, icHand;

        [Header("Task")]
        public string goal = "";

        const float PanelX = 0.37f, PanelZ = 1.0f, PanelScale = 0.00088f;
        Canvas _bg, _hud, _panelCanvas;
        RawImage _overlay;
        Texture2D _overlayTex;
        float _overlayA, _overlayTarget;
        string _lastImg;
        RectTransform _panel, _chat, _hudRoot, _targetRoot;
        CanvasGroup _panelGroup;
        ScrollRect _scroll;
        InputField _input;
        Text _status, _caption, _stepTitle, _stepTip, _stepIndex;
        CanvasGroup _captionGroup, _stepGroup;
        Image _stepBar, _orb, _orbRing, _orbGlow;
        Image[] _bars;
        RectTransform _thinking;
        bool _open, _busy, _speaking, _fakeListen;
        float _panelT, _speakLevel;
        JArray _steps;
        int _stepI = -1;
        readonly List<GameObject> _targets = new();

        // ---------------------------------------------------------------- build
        void Start()
        {
            HoloUI.Font = font;
            if (FindFirstObjectByType<EventSystem>() == null)
            {
                var es = new GameObject("EventSystem", typeof(EventSystem));
                es.AddComponent<UnityEngine.InputSystem.UI.InputSystemUIInputModule>();
            }
            BuildBackground();
            BuildHud();
            BuildPanel();
            wake.OnStatus += s => _status.text = s;
            if (viewer != null) viewer.OnStatus += s => ShowCaption(s);
            wake.OnWake += () => { if (!_open) SetOpen(true); ShowCaption("Listening…"); };
            wake.OnUtterance += t => { ShowCaption("“" + t + "”"); Send(t); };
            video.Play();
            SetOpen(false, instant: true);
            AddMuse("Hi, I'm Muse. I can see what you see. Ask me how to do something, what a part is, or to look something up on the web.", null);
        }

        Canvas MakeCanvas(string name, RenderMode mode, float plane, int order)
        {
            var go = new GameObject(name, typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            var c = go.GetComponent<Canvas>();
            c.renderMode = mode; c.worldCamera = cam; c.planeDistance = plane; c.sortingOrder = order;
            var s = go.GetComponent<CanvasScaler>();
            if (mode != RenderMode.WorldSpace)
            {
                s.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
                s.referenceResolution = new Vector2(1708, 960);
                s.matchWidthOrHeight = 0.5f;
            }
            else { s.dynamicPixelsPerUnit = 6f; s.referencePixelsPerUnit = 100; }
            return c;
        }

        void BuildBackground()
        {
            _bg = MakeCanvas("Passthrough", RenderMode.ScreenSpaceCamera, 30f, 0);
            var img = HoloUI.Rect("View", _bg.transform).Fill().gameObject.AddComponent<RawImage>();
            img.raycastTarget = false;
            video.display = img;
            var fit = img.gameObject.AddComponent<AspectRatioFitter>();
            fit.aspectMode = AspectRatioFitter.AspectMode.EnvelopeParent;
            fit.aspectRatio = 1708f / 960f;
            _overlay = HoloUI.Rect("Highlight", img.transform).Fill().gameObject.AddComponent<RawImage>();
            _overlay.raycastTarget = false;
            _overlay.color = new Color(1, 1, 1, 0);
            _overlayTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        }

        void BuildHud()
        {
            _hud = MakeCanvas("HUD", RenderMode.ScreenSpaceCamera, 0.5f, 3);
            _hudRoot = (RectTransform)_hud.transform;
            _targetRoot = HoloUI.Rect("Targets", _hudRoot).Fill();

            // status chip (top-left)
            var chip = HoloUI.Panel("Status", _hudRoot, HoloUI.Glass, 0.6f, frosted: true);
            chip.rectTransform.Anchor(new Vector2(0, 1), new Vector2(0, 1), new Vector2(0, 1), new Vector2(22, -22), new Vector2(380, 52));
            var dot = HoloUI.Rect("Dot", chip.transform).Anchor(new Vector2(0, 0.5f), new Vector2(0, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(24, 0), new Vector2(12, 12));
            var di = dot.gameObject.AddComponent<Image>(); di.sprite = HoloUI.Circle; di.color = HoloUI.Accent;
            var hl = HoloUI.Label("Holo", chip.transform, "<b>Holo</b>", 21, HoloUI.Ink);
            hl.rectTransform.Anchor(new Vector2(0, 0), new Vector2(0, 1), new Vector2(0, 0.5f), new Vector2(42, 0), new Vector2(70, 0));
            _status = HoloUI.Label("Text", chip.transform, "Say “Muse” to talk", 18, HoloUI.Sub);
            _status.rectTransform.Anchor(new Vector2(0, 0), new Vector2(1, 1), new Vector2(0, 0.5f), new Vector2(110, 0), new Vector2(-116, 0));

            // Muse orb (bottom centre) - tap to open
            var orbR = HoloUI.Rect("MuseOrb", _hudRoot).Anchor(new Vector2(0.5f, 0), new Vector2(0.5f, 0), new Vector2(0.5f, 0.5f), new Vector2(0, 64), new Vector2(74, 74));
            _orbGlow = HoloUI.Rect("Glow", orbR).Anchor(Vector2.one * 0.5f, Vector2.one * 0.5f, Vector2.one * 0.5f, Vector2.zero, new Vector2(120, 120)).gameObject.AddComponent<Image>();
            _orbGlow.sprite = HoloUI.Circle; _orbGlow.color = new Color(0.47f, 0.86f, 1f, 0.12f); _orbGlow.raycastTarget = false;
            _orb = orbR.gameObject.AddComponent<Image>(); _orb.sprite = HoloUI.Circle; _orb.color = new Color(1f, 1f, 1f, 1f); _orb.material = HoloUI.GlassMat;
            orbR.gameObject.AddComponent<Button>().onClick.AddListener(() => SetOpen(!_open));
            _orbRing = HoloUI.Rect("Ring", orbR).Fill(-3).gameObject.AddComponent<Image>();
            _orbRing.sprite = HoloUI.Ring; _orbRing.type = Image.Type.Sliced; _orbRing.pixelsPerUnitMultiplier = 0.27f; _orbRing.color = HoloUI.Accent; _orbRing.raycastTarget = false;
            _bars = new Image[5];
            for (int i = 0; i < 5; i++)
            {
                var b = HoloUI.Rect("Bar" + i, orbR).Anchor(Vector2.one * 0.5f, Vector2.one * 0.5f, Vector2.one * 0.5f, new Vector2((i - 2) * 9, 0), new Vector2(5, 10));
                _bars[i] = b.gameObject.AddComponent<Image>(); _bars[i].color = HoloUI.Ink; _bars[i].raycastTarget = false;
            }
            var hint = HoloUI.Label("Hint", orbR, "Say “Muse” · tap · M", 17, HoloUI.Sub, TextAnchor.UpperCenter);
            hint.rectTransform.Anchor(new Vector2(0.5f, 0), new Vector2(0.5f, 0), new Vector2(0.5f, 1), new Vector2(0, -8), new Vector2(260, 24));

            // live caption (above the orb)
            var cap = HoloUI.Panel("Caption", _hudRoot, HoloUI.Glass, 0.7f, frosted: true);
            cap.rectTransform.Anchor(new Vector2(0.5f, 0), new Vector2(0.5f, 0), new Vector2(0.5f, 0), new Vector2(0, 116), new Vector2(760, 52));
            _captionGroup = cap.gameObject.AddComponent<CanvasGroup>(); _captionGroup.alpha = 0; _captionGroup.blocksRaycasts = false;
            _caption = HoloUI.Label("Text", cap.transform, "", 19, HoloUI.Ink, TextAnchor.MiddleCenter);
            _caption.rectTransform.Fill(10);

            // step card (top centre)
            var card = HoloUI.Panel("StepCard", _hudRoot, HoloUI.Glass, 0.7f, frosted: true);
            card.rectTransform.Anchor(new Vector2(0, 1), new Vector2(0, 1), new Vector2(0, 1), new Vector2(22, -86), new Vector2(560, 104));   // top-left: keeps the middle of the view clear
            _stepGroup = card.gameObject.AddComponent<CanvasGroup>(); _stepGroup.alpha = 0;
            var badge = HoloUI.Rect("Badge", card.transform).Anchor(new Vector2(0, 0.5f), new Vector2(0, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(46, 6), new Vector2(46, 46));
            var bi = badge.gameObject.AddComponent<Image>(); bi.sprite = HoloUI.Ring; bi.type = Image.Type.Sliced; bi.pixelsPerUnitMultiplier = 0.36f; bi.color = HoloUI.Accent;
            _stepIndex = HoloUI.Label("N", badge, "1/3", 16, HoloUI.Ink, TextAnchor.MiddleCenter); _stepIndex.rectTransform.Fill();
            _stepTitle = HoloUI.Label("Title", card.transform, "", 21, HoloUI.Ink);
            _stepTitle.rectTransform.Anchor(new Vector2(0, 1), new Vector2(1, 1), new Vector2(0, 1), new Vector2(86, -14), new Vector2(-190, 34));
            _stepTip = HoloUI.Label("Tip", card.transform, "", 14, HoloUI.Sub);
            _stepTip.rectTransform.Anchor(new Vector2(0, 1), new Vector2(1, 1), new Vector2(0, 1), new Vector2(86, -52), new Vector2(-200, 26));
            var barBg = HoloUI.Rect("Bar", card.transform).Anchor(new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 0), new Vector2(22, 10), new Vector2(-44, 4));
            barBg.gameObject.AddComponent<Image>().color = new Color(0.24f, 0.35f, 0.5f, 0.6f);
            _stepBar = HoloUI.Rect("Fill", barBg).Anchor(Vector2.zero, new Vector2(0, 1), Vector2.zero, Vector2.zero, Vector2.zero).gameObject.AddComponent<Image>();
            _stepBar.color = HoloUI.Accent;
            var nav = HoloUI.Rect("Nav", card.transform).Anchor(new Vector2(1, 0.5f), new Vector2(1, 0.5f), new Vector2(1, 0.5f), new Vector2(-18, 6), new Vector2(150, 44));
            var hl2 = nav.gameObject.AddComponent<HorizontalLayoutGroup>(); hl2.spacing = 8; hl2.childAlignment = TextAnchor.MiddleRight; hl2.childControlWidth = false; hl2.childControlHeight = false;
            ((RectTransform)HoloUI.Pill("Back", nav, "Back", () => Step(-1), 16).transform).sizeDelta = new Vector2(70, 38);
            ((RectTransform)HoloUI.Pill("Next", nav, "Next →", () => Step(+1), 16, new Color(1f, 1f, 1f, 0.18f)).transform).sizeDelta = new Vector2(78, 38);
        }

        void BuildPanel()
        {
            _panelCanvas = MakeCanvas("MusePanel", RenderMode.WorldSpace, 0, 2);
            var t = _panelCanvas.transform;
            t.SetParent(cam.transform, false);
            t.localPosition = new Vector3(PanelX, 0.02f, PanelZ);
            t.localRotation = Quaternion.Euler(0, 11f, 0);
            t.localScale = Vector3.one * PanelScale;
            var root = (RectTransform)t;
            root.sizeDelta = new Vector2(640, 780);
            _panelGroup = _panelCanvas.gameObject.AddComponent<CanvasGroup>();

            var bg = HoloUI.Panel("Window", t, HoloUI.Glass, 0.7f, frosted: true);
            _panel = bg.rectTransform.Fill();

            // header
            var dot = HoloUI.Rect("Dot", _panel).Anchor(new Vector2(0, 1), new Vector2(0, 1), new Vector2(0.5f, 0.5f), new Vector2(34, -38), new Vector2(16, 16));
            var di = dot.gameObject.AddComponent<Image>(); di.sprite = HoloUI.Circle; di.color = HoloUI.Accent;
            var title = HoloUI.Label("Title", _panel, "<b>Muse</b>", 30, HoloUI.Ink);
            title.rectTransform.Anchor(new Vector2(0, 1), new Vector2(0, 1), new Vector2(0, 0.5f), new Vector2(52, -38), new Vector2(200, 36));
            var sub = HoloUI.Label("Sub", _panel, "sees your view · Muse Spark", 18, HoloUI.Sub);
            sub.rectTransform.Anchor(new Vector2(0, 1), new Vector2(0, 1), new Vector2(0, 0.5f), new Vector2(140, -40), new Vector2(300, 30));
            var close = HoloUI.IconButton("Close", _panel, icClose, () => SetOpen(false), 40, new Color(1, 1, 1, 0.08f));
            ((RectTransform)close.transform).Anchor(new Vector2(1, 1), new Vector2(1, 1), new Vector2(0.5f, 0.5f), new Vector2(-38, -38), new Vector2(40, 40));
            var line = HoloUI.Rect("Line", _panel).Anchor(new Vector2(0, 1), new Vector2(1, 1), new Vector2(0.5f, 1), new Vector2(0, -74), new Vector2(-40, 1));
            line.gameObject.AddComponent<Image>().color = new Color(0.63f, 0.82f, 1f, 0.18f);

            // ask / search bar
            var bar = HoloUI.Panel("AskBar", _panel, new Color(1f, 1f, 1f, 0.10f), 0.9f);
            bar.rectTransform.Anchor(new Vector2(0, 0), new Vector2(1, 0), new Vector2(0.5f, 0), new Vector2(0, 22), new Vector2(-40, 62));
            var ic = HoloUI.Icon("Search", bar.transform, icSearch, HoloUI.Sub, 26);
            ic.rectTransform.Anchor(new Vector2(0, 0.5f), new Vector2(0, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(30, 0), new Vector2(26, 26));
            var inR = HoloUI.Rect("Input", bar.transform).Anchor(new Vector2(0, 0), new Vector2(1, 1), new Vector2(0, 0.5f), new Vector2(54, 0), new Vector2(-170, 0));
            inR.gameObject.AddComponent<Image>().color = new Color(0, 0, 0, 0);
            _input = inR.gameObject.AddComponent<InputField>();
            var txt = HoloUI.Label("Text", inR, "", 22, HoloUI.Ink); txt.rectTransform.Fill(); txt.supportRichText = false;
            var ph = HoloUI.Label("Placeholder", inR, "Ask or search…  or just say “Muse”", 21, new Color(0.6f, 0.72f, 0.86f, 0.75f));
            ph.rectTransform.Fill();
            _input.textComponent = txt; _input.placeholder = ph; _input.lineType = InputField.LineType.SingleLine;
            _input.onSubmit.AddListener(s => { if (!string.IsNullOrWhiteSpace(s)) { Send(s); _input.text = ""; _input.ActivateInputField(); } });
            var mic = HoloUI.IconButton("Mic", bar.transform, icMic, () => wake.BeginPushToTalk(), 46, new Color(0.47f, 0.86f, 1f, 0.22f));
            ((RectTransform)mic.transform).Anchor(new Vector2(1, 0.5f), new Vector2(1, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(-86, 0), new Vector2(46, 46));
            var send = HoloUI.IconButton("Send", bar.transform, icSend, () => { if (!string.IsNullOrWhiteSpace(_input.text)) { Send(_input.text); _input.text = ""; } }, 46, HoloUI.Accent * new Color(1, 1, 1, 0.9f));
            ((RectTransform)send.transform).Anchor(new Vector2(1, 0.5f), new Vector2(1, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(-34, 0), new Vector2(46, 46));

            // quick actions
            var quick = HoloUI.Rect("Quick", _panel).Anchor(new Vector2(0, 0), new Vector2(1, 0), new Vector2(0.5f, 0), new Vector2(0, 96), new Vector2(-40, 44));
            var qh = quick.gameObject.AddComponent<HorizontalLayoutGroup>(); qh.spacing = 8; qh.childControlWidth = false; qh.childControlHeight = false; qh.childAlignment = TextAnchor.MiddleLeft;
            foreach (var q in new[] { "How do I connect this?", "What is this?", "Search: is this port USB 3?" })
            {
                var qq = q;
                var b = HoloUI.Pill("Q", quick, q, () => Send(qq.Replace("Search: ", "")), 17);
                var le = b.GetComponent<LayoutElement>();
                ((RectTransform)b.transform).sizeDelta = new Vector2(le.preferredWidth, 38);
            }

            // chat scroll
            var view = HoloUI.Rect("Chat", _panel).Anchor(Vector2.zero, Vector2.one, new Vector2(0.5f, 0.5f), new Vector2(0, 36), new Vector2(-28, -232));
            view.gameObject.AddComponent<RectMask2D>();
            _scroll = view.gameObject.AddComponent<ScrollRect>();
            _scroll.horizontal = false; _scroll.movementType = ScrollRect.MovementType.Clamped; _scroll.scrollSensitivity = 30;
            view.gameObject.AddComponent<Image>().color = new Color(0, 0, 0, 0);
            _chat = HoloUI.Rect("Content", view).Anchor(new Vector2(0, 1), new Vector2(1, 1), new Vector2(0.5f, 1), Vector2.zero, Vector2.zero);
            var vl = _chat.gameObject.AddComponent<VerticalLayoutGroup>();
            vl.spacing = 12; vl.padding = new RectOffset(6, 6, 6, 12); vl.childControlHeight = true; vl.childControlWidth = true; vl.childForceExpandHeight = false;
            _chat.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            _scroll.content = _chat; _scroll.viewport = view;
        }

        // ---------------------------------------------------------------- chat bubbles
        RectTransform Bubble(bool user, Color fill, out RectTransform body)
        {
            var row = HoloUI.Rect(user ? "You" : "Muse", _chat);
            var h = row.gameObject.AddComponent<HorizontalLayoutGroup>();
            h.padding = new RectOffset(user ? 90 : 0, user ? 0 : 40, 0, 0);
            h.childControlHeight = true; h.childControlWidth = true; h.childForceExpandWidth = true;
            var b = HoloUI.Panel("Bubble", row, fill, 0.55f, outline: !user);
            var v = b.gameObject.AddComponent<VerticalLayoutGroup>();
            v.padding = new RectOffset(18, 18, 14, 14); v.spacing = 8; v.childControlHeight = true; v.childControlWidth = true; v.childForceExpandHeight = false;
            body = b.rectTransform;
            return row;
        }

        Text Line(RectTransform body, string text, int size, Color color, FontStyle style = FontStyle.Normal)
        {
            var t = HoloUI.Label("Line", body, text, size, color, TextAnchor.UpperLeft, style);
            return t;
        }

        void AddUser(string text)
        {
            Bubble(true, HoloUI.UserBubble, out var body);
            Line(body, text, 24, HoloUI.Ink);
            ScrollDown();
        }

        RectTransform AddMuse(string text, string meta)
        {
            var row = Bubble(false, new Color(1f, 1f, 1f, 0.06f), out var body);
            Line(body, text, 24, HoloUI.Ink);
            if (!string.IsNullOrEmpty(meta)) Line(body, meta, 16, HoloUI.Sub);
            ScrollDown();
            return body;
        }

        void ShowThinking(string label)
        {
            Bubble(false, new Color(1f, 1f, 1f, 0.04f), out var body);
            _thinking = (RectTransform)body.parent;
            Line(body, label, 20, HoloUI.Sub).gameObject.AddComponent<Dots>();
            ScrollDown();
        }

        void ClearThinking() { if (_thinking) Destroy(_thinking.gameObject); _thinking = null; }

        void ScrollDown() => StartCoroutine(ScrollNext());
        IEnumerator ScrollNext() { yield return null; Canvas.ForceUpdateCanvases(); _scroll.verticalNormalizedPosition = 0; }

        // ---------------------------------------------------------------- talk to Muse
        /// <summary>Scripted demo: plays the "listening" state with the words appearing, then sends. (Recording aid.)</summary>
        public IEnumerator Say(string text, float seconds = 2.6f)
        {
            if (!_open) SetOpen(true);
            _fakeListen = true;
            var words = text.Split(' ');
            float t0 = Time.time;
            while (Time.time - t0 < seconds)
            {
                int n = Mathf.Clamp(Mathf.CeilToInt((Time.time - t0) / seconds * words.Length), 1, words.Length);
                _caption.text = "\u201c" + string.Join(" ", words, 0, n) + "\u201d"; _capUntil = Time.time + 2.5f;
                yield return null;
            }
            _fakeListen = false;
            Send(text);
        }

        public bool Busy => _busy;
        public bool Speaking => _speaking;
        public static Action<AudioClip> OnSpeak;

        public void Send(string text)
        {
            if (_busy) return;
            if (!_open) SetOpen(true);
            text = text.Trim();
            AddUser(text);
            if (string.IsNullOrEmpty(goal)) goal = text;
            _busy = true;
            if (System.Text.RegularExpressions.Regex.IsMatch(text, @"\b(open|show|load)\b.*\b(last|my|saved)\b.*\b(build|design|model|drone)\b", System.Text.RegularExpressions.RegexOptions.IgnoreCase))
            {
                ShowThinking("Opening your saved build");
                StartCoroutine(OpenLastBuild());
                return;
            }
            if (System.Text.RegularExpressions.Regex.IsMatch(text, @"\b(build|make|design|3d print|print|create|assemble)\b.*\b(a|an|my|me|this)\b", System.Text.RegularExpressions.RegexOptions.IgnoreCase)
                && !System.Text.RegularExpressions.Regex.IsMatch(text, @"\b(connect|plug)\b", System.Text.RegularExpressions.RegexOptions.IgnoreCase))
            {
                StartCoroutine(BuildFlow(text));
                return;
            }
            bool searchy = System.Text.RegularExpressions.Regex.IsMatch(text, @"\b(search|look up|web|usb ?3|spec|charge|compatible|price|manual)\b", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            ShowThinking(searchy ? "Searching the web" : "Looking at your view");
            video.Pause();                     // freeze the view Muse is reasoning about (boxes stay valid)
            var img = video.GrabBase64();
            _lastImg = img;
            ClearTargets();
            float t0 = Time.time;
            StartCoroutine(api.Chat(img, text, goal, r => { ClearThinking(); _busy = false; Show(r, Time.time - t0); },
                e => { ClearThinking(); _busy = false; AddMuse("I couldn't reach Muse (" + e.Split('\n')[0] + "). Is server.py running?", null); }));
        }

        void Show(JObject r, float secs)
        {
            var kind = (string)r["kind"];
            var meta = $"Muse Spark · {secs:0.0} s";
            if (kind == "plan")
            {
                var body = AddMuse((string)r["summary"] ?? "Here's the plan.", (string)r["identified"] is string id ? "I see: " + id : null);
                _steps = r["steps"] as JArray ?? new JArray();
                int i = 1;
                foreach (JObject s in _steps)
                {
                    var l = Line(body, $"<b>{i++}.</b>  {(string)s["instruction"]}", 22, HoloUI.Ink);
                }
                var start = HoloUI.Pill("Start", body, "Start guide  →", () => StartGuide(), 20, new Color(1f, 1f, 1f, 0.16f));
                Line(body, meta, 16, HoloUI.Sub);
            }
            else if (kind == "search")
            {
                var bq = AddMuse("<color=#96c8f5>Searched the web for</color> “" + (string)r["query"] + "”", null);
                var body = AddMuse((string)r["answer"] ?? "", null);
                foreach (JObject s in (r["sources"] as JArray) ?? new JArray())
                {
                    var url = (string)s["url"] ?? "";
                    var host = Uri.TryCreate(url, UriKind.Absolute, out var u) ? u.Host.Replace("www.", "") : url;
                    var src = HoloUI.Pill("Src", body, $"[{(int?)s["n"]}]  {Trim((string)s["title"], 30)}   · {host}", () => Application.OpenURL(url), 17, new Color(0.1f, 0.17f, 0.29f, 0.95f));
                    src.GetComponentInChildren<Text>().alignment = TextAnchor.MiddleLeft;
                }
                Line(body, meta + " + web", 16, HoloUI.Sub);
            }
            else
            {
                var body = AddMuse((string)r["answer"] ?? "", r["resource"] is JValue rv && rv.Type == JTokenType.String ? "Learn: " + (string)rv : null);
                Line(body, meta, 16, HoloUI.Sub);
                var tl = (string)r["target_label"];
                if (!string.IsNullOrEmpty(tl)) Highlight(tl, "muse", r["target_box"] as JArray);
                else if (r["target_box"] is JArray tb && tb.Count == 4) Target(tb, "here", 7f);
            }
            var say = (string)r["say"];
            if (!string.IsNullOrWhiteSpace(say)) StartCoroutine(Speak(say));
            ScrollDown();
        }

        JObject _build;
        public int BuildStage;
        public static bool Recording;       // 0 none, 1 design, 2 render, 3 3D model ready (recording aid)
        RectTransform _toolbar, _buildActions;
        RawImage _buildImg;
        Text _buildStatus;

        /// <summary>Build mode, staged so the user sees progress: Muse's design (~10 s) -> render (~1 min) -> 3D model.</summary>
        IEnumerator BuildFlow(string text)
        {
            BuildStage = 0;
            ShowThinking("Muse is designing it");
            var im = video.GrabBase64();
            float t0 = Time.time;
            JObject r = null; string err = null;
            yield return api.BuildDesign(im, text, x => r = x, e => err = e);
            ClearThinking();
            if (r == null) { _busy = false; AddMuse("Build failed: " + err?.Split('\n')[0], null); yield break; }
            ShowBuild(r, Time.time - t0);
            BuildStage = 1;
            var d = r["design"] as JObject;
            StartCoroutine(Speak($"Here's a design for a {(string)d["name"]}. I'm rendering it now, then I'll make a 3D model you can print."));
            var id = (string)r["id"];

            _buildStatus.text = "Rendering the finished build…";
            JObject ri = null; err = null;
            yield return api.BuildRender(id, x => ri = x, e => err = e);
            if (ri == null) { _buildStatus.text = "Render unavailable: " + err?.Split('\n')[0]; _busy = false; yield break; }
            _build["image"] = ri["image"];
            SetBuildImage((string)ri["image"]);
            BuildStage = 2;
            _buildStatus.text = "Turning the render into a 3D model…";

            JObject rm = null; err = null;
            yield return api.BuildMesh(id, x => rm = x, e => err = e);
            _busy = false;
            if (rm == null) { _buildStatus.text = "3D model unavailable: " + err?.Split('\n')[0]; yield break; }
            foreach (var k in new[] { "glb", "stl_path", "glb_path", "mesh" }) _build[k] = rm[k];
            BuildReady(Time.time - t0);
            BuildStage = 3;
            StartCoroutine(Speak("Your 3D model is ready. Pinch it to move it, use two hands to scale, or export the S.T.L. to print."));
        }

        IEnumerator OpenLastBuild()
        {
            JObject list = null;
            using (var req = UnityEngine.Networking.UnityWebRequest.Get(api.serverUrl + "/api/builds"))
            {
                yield return req.SendWebRequest();
                if (req.result == UnityEngine.Networking.UnityWebRequest.Result.Success) list = JObject.Parse(req.downloadHandler.text);
            }
            var first = (list?["builds"] as JArray)?.FirstOrDefault(b => (bool?)b["has_model"] == true);
            if (first == null) { ClearThinking(); _busy = false; AddMuse("You don't have a saved build yet. Say \u201cMuse, help me build \u2026\u201d", null); yield break; }
            JObject r = null;
            yield return api.GetBuild((string)first["id"], x => r = x, e => { });
            ClearThinking(); _busy = false;
            if (r == null) yield break;
            ShowBuild(r, 0);
            if (r["image"] != null) SetBuildImage((string)r["image"]);
            if (r["glb"] != null) BuildReady(0);
            View3D();
        }

        void SetBuildImage(string b64)
        {
            var tex = new Texture2D(2, 2);
            tex.LoadImage(Convert.FromBase64String(b64));
            _buildImg.texture = tex;
            _buildImg.color = Color.white;
        }

        void ShowBuild(JObject r, float secs)
        {
            _build = r;
            var d = r["design"] as JObject ?? new JObject();
            var body = AddMuse($"<b>{(string)d["name"]}</b> \u2014 {(string)d["summary"]}", null);
            var slot = HoloUI.Rect("RenderSlot", body);                       // fixed square slot: layout-safe
            var le = slot.gameObject.AddComponent<LayoutElement>(); le.preferredHeight = 280; le.minHeight = 280;
            _buildImg = HoloUI.Rect("Render", slot).gameObject.AddComponent<RawImage>();
            var rt = _buildImg.rectTransform;
            rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0, 0.5f);
            rt.anchoredPosition = Vector2.zero; rt.sizeDelta = new Vector2(280, 280);
            _buildImg.color = new Color(1, 1, 1, 0.06f);
            _buildStatus = Line(body, "", 15, new Color(0.72f, 0.82f, 1f));
            _buildStatus.gameObject.AddComponent<Dots>();
            var parts = d["parts"] as JArray;
            if (parts != null) Line(body, "<b>Parts</b>  " + string.Join(" \u00b7 ", parts.Select(p => $"{(int?)p["qty"] ?? 1}\u00d7 {(string)p["name"]}")), 16, HoloUI.Sub);
            var pr = d["printable"] as JArray;
            if (pr != null && pr.Count > 0) Line(body, "<b>3D print</b>  " + string.Join(" \u00b7 ", pr.Select(p => (string)p["name"])), 16, HoloUI.Sub);
            var wiring = d["wiring"] as JArray;
            if (wiring != null && wiring.Count > 0) Line(body, "<b>Wiring</b>\n" + string.Join("\n", wiring.Take(6).Select(w => "\u2022 " + (string)w)), 15, HoloUI.Ink);
            var code = d["code"] as JObject;
            if (code != null) Line(body, $"<b>Code</b>  {(string)code["filename"]} ({(string)code["language"]}) saved with the build", 15, HoloUI.Sub);
            if (d["safety"] != null) Line(body, "<b>Safety</b>  " + (string)d["safety"], 14, HoloUI.Sub);
            _buildActions = HoloUI.Rect("Actions", body);
            var h = _buildActions.gameObject.AddComponent<HorizontalLayoutGroup>(); h.spacing = 8; h.childControlWidth = false; h.childControlHeight = false;
            _buildActions.gameObject.AddComponent<LayoutElement>().preferredHeight = 44;
            _steps = d["steps"] as JArray;
            if (_steps != null && _steps.Count > 0) Size(HoloUI.Pill("Guide", _buildActions, "Start guide \u2192", () => StartGuide(), 17), 140);
            ScrollDown();
        }

        void BuildReady(float secs)
        {
            if (_buildStatus) { Destroy(_buildStatus.GetComponent<Dots>()); var m = _build["mesh"] as JObject;
                _buildStatus.text = "Muse Spark design \u00b7 Z-Image render \u00b7 Hunyuan3D mesh" + (secs > 0 ? $" \u00b7 {secs:0} s" : "")
                    + (m != null ? $" \u00b7 {(int)m["faces"] / 1000}k faces{((bool)m["watertight"] ? " \u00b7 watertight, printable" : "")}" : ""); }
            var v = HoloUI.Pill("View3D", _buildActions, "View in 3D", () => View3D(), 17, new Color(1, 1, 1, 0.18f));
            v.transform.SetSiblingIndex(0); Size(v, 140);
            var st = HoloUI.Pill("STL", _buildActions, "Export STL", () => ExportStl(), 17);
            st.transform.SetSiblingIndex(1); Size(st, 130);
            ScrollDown();
        }

        static void Size(Button b, float w) { ((RectTransform)b.transform).sizeDelta = new Vector2(w, 40); }

        public void View3D()
        {
            if (_build?["glb"] == null || viewer == null) return;
            SetOpen(false);
            StartCoroutine(viewer.Show(Convert.FromBase64String((string)_build["glb"])));
            BuildToolbar();
        }

        public void ExportStl()
        {
            var path = (string)_build?["stl_path"];
            if (string.IsNullOrEmpty(path)) return;
            ShowCaption("Watertight STL saved \u2014 open it in your slicer (Cura, PrusaSlicer, Bambu Studio) and print");
            if (!Recording) Application.OpenURL("file://" + System.IO.Path.GetDirectoryName(path));
        }

        void BuildToolbar()
        {
            if (_toolbar != null) { _toolbar.gameObject.SetActive(true); return; }
            var bar = HoloUI.Panel("ModelBar", _hudRoot, HoloUI.Glass, 0.7f, frosted: true);
            _toolbar = bar.rectTransform.Anchor(new Vector2(0.5f, 0), new Vector2(0.5f, 0), new Vector2(0.5f, 0), new Vector2(0, 112), new Vector2(720, 58));
            var h = HoloUI.Rect("Row", _toolbar).Fill(9).gameObject.AddComponent<HorizontalLayoutGroup>();
            h.spacing = 8; h.childControlWidth = false; h.childControlHeight = false; h.childAlignment = TextAnchor.MiddleCenter;
            Size(HoloUI.Pill("Follow", h.transform, "Follow / Pin", () => viewer.ToggleFollow(), 16), 120);
            Size(HoloUI.Pill("Top", h.transform, "Top view", () => viewer.TopView(), 16), 100);
            Size(HoloUI.Pill("Spin", h.transform, "Spin", () => viewer.ToggleSpin(), 16), 80);
            Size(HoloUI.Pill("Reset", h.transform, "Reset", () => viewer.ResetView(), 16), 80);
            Size(HoloUI.Pill("BarSTL", h.transform, "Export STL", () => ExportStl(), 16), 120);
            Size(HoloUI.Pill("Close", h.transform, "Close", () => { viewer.Clear(); _toolbar.gameObject.SetActive(false); }, 16), 80);
        }

        /// <summary>Short, easy step text: first clause, at most n words.</summary>
        static string Short(string t, int n)
        {
            if (string.IsNullOrEmpty(t)) return "";
            t = System.Text.RegularExpressions.Regex.Split(t, @"[;,(\u2014]| then ")[0].Trim().TrimEnd('.');
            var w = t.Split(' ');
            return w.Length <= n ? t : string.Join(" ", w, 0, n) + "\u2026";
        }

        static string Trim(string s, int n) => string.IsNullOrEmpty(s) || s.Length <= n ? s : s.Substring(0, n - 1) + "…";

        IEnumerator Speak(string text)
        {
            JObject res = null;
            yield return api.Tts(text, r => res = r, e => Debug.LogWarning(e));
            if (res == null) yield break;
            var clip = WavToClip(Convert.FromBase64String((string)res["audio"]));
            if (clip == null) yield break;
            voice.clip = clip; voice.Play();
            OnSpeak?.Invoke(clip);
            _speaking = true;
            ShowCaption(text);
            while (voice.isPlaying) yield return null;
            _speaking = false;
        }

        static AudioClip WavToClip(byte[] wav)
        {
            if (wav.Length < 44) return null;
            int ch = BitConverter.ToInt16(wav, 22), rate = BitConverter.ToInt32(wav, 24);
            int pos = 12;
            while (pos + 8 < wav.Length && !(wav[pos] == 'd' && wav[pos + 1] == 'a' && wav[pos + 2] == 't' && wav[pos + 3] == 'a'))
                pos += 8 + BitConverter.ToInt32(wav, pos + 4);
            int len = Math.Min(BitConverter.ToInt32(wav, pos + 4), wav.Length - pos - 8) / 2;
            var data = new float[len];
            for (int i = 0; i < len; i++) data[i] = BitConverter.ToInt16(wav, pos + 8 + i * 2) / 32768f;
            var clip = AudioClip.Create("muse", len / ch, ch, rate, false);
            clip.SetData(data, 0);
            return clip;
        }

        // ---------------------------------------------------------------- guide
        public void StartGuide()
        {
            if (_steps == null || _steps.Count == 0) return;
            _stepI = -1;
            SetOpen(false);
            Step(+1);
        }

        void Step(int d)
        {
            if (_steps == null || _steps.Count == 0) return;
            _stepI = Mathf.Clamp(_stepI + d, 0, _steps.Count);
            ClearTargets();
            if (_stepI >= _steps.Count)
            {
                _stepIndex.text = "✓"; _stepTitle.text = "Done"; _stepTip.text = "Muse can check it: say “Muse, is it right?”";
                _stepBar.rectTransform.anchorMax = new Vector2(1, 1);
                StartCoroutine(Speak("All steps done."));
                return;
            }
            var s = (JObject)_steps[_stepI];
            _stepIndex.text = $"{_stepI + 1}/{_steps.Count}";
            _stepTitle.text = Short((string)s["instruction"], 7);
            _stepTip.text = Short((string)s["learn"], 9);
            _stepBar.rectTransform.anchorMax = new Vector2((_stepI + 1f) / _steps.Count, 1);
            // highlight only this step's object, coloured by its role (amber = grab it, magenta = where it goes)
            var label = (string)s["target_label"];
            string role = (s["place"]?.Type == JTokenType.String || s["connect"] is JArray) ? "dest" : (s["pick"]?.Type == JTokenType.String ? "grab" : "muse");
            if (!string.IsNullOrEmpty(label)) Highlight(label, role, s["target_box"] as JArray);
            StartCoroutine(Speak((string)s["instruction"]));
        }

        void ClearTargets() { foreach (var g in _targets) if (g) Destroy(g); _targets.Clear(); _overlayTarget = 0; }

        /// <summary>SAM 3.1 mask of exactly this thing, drawn as a glow in the role colour. Falls back to Muse's box.</summary>
        void Highlight(string label, string role, JArray fallbackBox)
        {
            if (string.IsNullOrEmpty(_lastImg)) return;
            StartCoroutine(api.Segment(_lastImg, label, role, r =>
            {
                if (r.Value<bool>("found"))
                {
                    _overlayTex.LoadImage(Convert.FromBase64String((string)r["overlay"]));
                    _overlay.texture = _overlayTex;
                    _overlayTarget = 1;
                }
                else if (fallbackBox != null && fallbackBox.Count == 4) Target(fallbackBox, label, 7f);
            }, e => { if (fallbackBox != null && fallbackBox.Count == 4) Target(fallbackBox, label, 7f); }));
        }

        /// <summary>Corner brackets + label on a 0-1000 box of the frozen frame.</summary>
        void Target(JArray box, string label, float life)
        {
            var area = ((RectTransform)video.display.transform).rect;   // background rect (same reference size)
            var hud = _hudRoot.rect;
            float x1 = (float)box[0] / 1000f, y1 = (float)box[1] / 1000f, x2 = (float)box[2] / 1000f, y2 = (float)box[3] / 1000f;
            // background envelopes the screen: map image space -> HUD space with the same envelope
            float s = Mathf.Max(hud.width / 1708f, hud.height / 960f);
            float w = 1708f * s, h = 960f * s;
            var min = new Vector2(-w / 2 + x1 * w, h / 2 - y2 * h);
            var max = new Vector2(-w / 2 + x2 * w, h / 2 - y1 * h);
            var r = HoloUI.Rect("Target", _targetRoot).Anchor(Vector2.one * 0.5f, Vector2.one * 0.5f, Vector2.one * 0.5f, (min + max) / 2, max - min);
            var ring = r.gameObject.AddComponent<Image>(); ring.sprite = HoloUI.Ring; ring.type = Image.Type.Sliced; ring.pixelsPerUnitMultiplier = 0.5f;
            ring.color = HoloUI.Accent; ring.raycastTarget = false;
            r.gameObject.AddComponent<Pulse>();
            if (!string.IsNullOrEmpty(label))
            {
                var tag = HoloUI.Panel("Tag", r, HoloUI.Glass, 0.5f, frosted: true);
                var tl = HoloUI.Label("T", tag.transform, label, 15, HoloUI.Ink, TextAnchor.MiddleCenter);
                tag.rectTransform.Anchor(new Vector2(0.5f, 1), new Vector2(0.5f, 1), new Vector2(0.5f, 0), new Vector2(0, 8), new Vector2(tl.preferredWidth + 28, 32));
                tl.rectTransform.Fill();
            }
            _targets.Add(r.gameObject);
            if (life > 0) Destroy(r.gameObject, life);
        }

        // ---------------------------------------------------------------- panel / orb / captions
        void SetOpen(bool open, bool instant = false)
        {
            _open = open;
            _panelGroup.interactable = open; _panelGroup.blocksRaycasts = open;
            if (instant) { _panelT = open ? 1 : 0; ApplyPanel(); }
            if (open) _input.ActivateInputField();
        }

        void ApplyPanel()
        {
            float e = 1 - Mathf.Pow(1 - _panelT, 3);
            _panelGroup.alpha = e;
            _panelCanvas.transform.localScale = Vector3.one * PanelScale * Mathf.Lerp(0.94f, 1f, e);
            _panelCanvas.transform.localPosition = new Vector3(PanelX, Mathf.Lerp(-0.03f, 0.0f, e), PanelZ);
        }

        float _capUntil, _lastTap = -1;
        void ShowCaption(string t) { _caption.text = t; _capUntil = Time.time + Mathf.Clamp(t.Length * 0.06f, 2.5f, 7f); }

        void Update()
        {
            var kb = UnityEngine.InputSystem.Keyboard.current;
            bool typing = _input != null && _input.isFocused;
            if (kb != null)
            {
                if (kb.escapeKey.wasPressedThisFrame) SetOpen(false);
                if (!typing)
                {
                    if (kb.mKey.wasPressedThisFrame || kb.tabKey.wasPressedThisFrame) SetOpen(!_open);
                    if (kb.spaceKey.wasPressedThisFrame) video.Toggle();
                    if (kb.nKey.wasPressedThisFrame) Step(+1);
                    if (kb.bKey.wasPressedThisFrame) Step(-1);
                }
            }

            // touch: double-tap anywhere on the view summons Muse (pinch on Quest)
            var mouse = UnityEngine.InputSystem.Mouse.current;
            if (mouse != null && mouse.leftButton.wasPressedThisFrame && !EventSystem.current.IsPointerOverGameObject())
            {
                if (Time.time - _lastTap < 0.35f) { SetOpen(!_open); _lastTap = -1; }
                else _lastTap = Time.time;
            }

            _panelT = Mathf.MoveTowards(_panelT, _open ? 1 : 0, Time.deltaTime * 4.5f);
            ApplyPanel();

            // orb
            float t = Time.time;
            bool listening = wake.Current == WakeListener.State.Capturing || _fakeListen;
            bool thinking = _busy || wake.Current == WakeListener.State.Transcribing;
            if (_speaking && voice.isPlaying)
            {
                var buf = new float[256];
                voice.GetOutputData(buf, 0);
                float m = 0; foreach (var v in buf) m = Mathf.Max(m, Mathf.Abs(v));
                _speakLevel = Mathf.Lerp(_speakLevel, m, 0.5f);
            }
            else _speakLevel = Mathf.Lerp(_speakLevel, 0, 0.2f);
            float level = _fakeListen ? 0.35f + 0.25f * Mathf.Sin(t * 9) : listening ? wake.Level : _speakLevel;
            for (int i = 0; i < _bars.Length; i++)
            {
                float hgt = listening || _speaking ? 8 + 26 * level * (0.6f + 0.4f * Mathf.Sin(t * 14 + i * 1.7f)) : 6 + 2 * Mathf.Sin(t * 2 + i);
                _bars[i].rectTransform.sizeDelta = new Vector2(5, hgt);
                _bars[i].color = new Color(1, 1, 1, thinking ? 0.25f : 0.95f);
            }
            _orbRing.transform.localRotation = thinking ? Quaternion.Euler(0, 0, -t * 220) : Quaternion.identity;
            _orbRing.fillAmount = 1;
            _orbRing.color = new Color(0.47f, 0.86f, 1f, listening ? 1f : thinking ? 0.85f : 0.45f + 0.25f * Mathf.Sin(t * 2));
            float g = listening || _speaking ? 1.15f + level * 0.8f : 1f + 0.05f * Mathf.Sin(t * 2);
            _orbGlow.transform.localScale = Vector3.one * g;
            _orbGlow.color = new Color(0.47f, 0.86f, 1f, listening || _speaking ? 0.28f : 0.1f);

            // caption + step card fades
            // highlight overlay: fades in on a result, out when the view moves on (it belongs to the frozen frame)
            if (video.IsPlaying) _overlayTarget = 0;
            float pul = _overlayTarget > 0 ? 0.85f + 0.15f * Mathf.Sin(t * 4.5f) : 1f;
            _overlayA = Mathf.MoveTowards(_overlayA, _overlayTarget, Time.deltaTime * 3f);
            _overlay.color = new Color(1, 1, 1, _overlayA * pul);
            _captionGroup.alpha = Mathf.MoveTowards(_captionGroup.alpha, Time.time < _capUntil ? 1 : 0, Time.deltaTime * 3);
            _stepGroup.alpha = Mathf.MoveTowards(_stepGroup.alpha, _stepI >= 0 ? 1 : 0, Time.deltaTime * 3);
        }
    }

    /// <summary>Animated "…" after a label.</summary>
    public class Dots : MonoBehaviour
    {
        Text _t; string _base;
        void Start() { _t = GetComponent<Text>(); _base = _t.text; }
        void Update()
        {
            var cur = _t.text.TrimEnd('.', '\u2026');
            if (cur != _base.TrimEnd('.', '\u2026')) _base = cur;      // text was changed by the app: follow it
            int n = (int)(Time.time * 3) % 4; _t.text = _base.TrimEnd('.', '\u2026') + new string('.', n);
        }
    }

    /// <summary>Gentle pulse for target highlights.</summary>
    public class Pulse : MonoBehaviour
    {
        Image _i;
        void Start() => _i = GetComponent<Image>();
        void Update()
        {
            float s = 1 + 0.03f * Mathf.Sin(Time.time * 5);
            transform.localScale = new Vector3(s, s, 1);
            if (_i) _i.color = new Color(_i.color.r, _i.color.g, _i.color.b, 0.7f + 0.3f * Mathf.Sin(Time.time * 5));
        }
    }
}
