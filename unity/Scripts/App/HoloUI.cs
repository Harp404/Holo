using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace Holo
{
    /// <summary>Tiny UI kit in the Meta Horizon style: rounded glass panels, Inter text, pill buttons.</summary>
    public static class HoloUI
    {
        public static Font Font;
        // Horizon OS / glassmorphism palette: frosted panels (Holo/UIGlass), white text, faint white edges
        public static readonly Color Glass = new(1f, 1f, 1f, 1f);                  // opacity of a frosted panel
        public static readonly Color GlassHi = new(1f, 1f, 1f, 0.14f);             // plain veil (buttons / chips)
        public static readonly Color UserBubble = new(1f, 1f, 1f, 0.13f);
        public static readonly Color Edge = new(1f, 1f, 1f, 0.36f);
        public static readonly Color Ink = new(0.97f, 0.98f, 1f, 1f);
        public static readonly Color Sub = new(0.80f, 0.85f, 0.92f, 1f);
        public static readonly Color Accent = new(0.45f, 0.76f, 1f, 1f);

        static Material _glassMat;
        public static Material GlassMat
        {
            get
            {
                if (_glassMat == null)
                {
                    var sh = Shader.Find("Holo/UIGlass");
                    if (sh != null)
                    {
                        _glassMat = new Material(sh);
                        _glassMat.SetFloat("_Blur", 8f);
                        _glassMat.SetFloat("_Darken", 0.48f);
                        _glassMat.SetFloat("_Veil", 0.05f);
                    }
                }
                return _glassMat;
            }
        }

        static Sprite _round, _ring, _circle;

        /// <summary>9-sliced rounded rect, built once (radius 32 px, antialiased).</summary>
        public static Sprite Round => _round ??= MakeRound(128, 32, false);
        public static Sprite Ring => _ring ??= MakeRound(128, 32, true);
        public static Sprite Circle => _circle ??= MakeCircle(128);

        static Sprite MakeRound(int size, int r, bool ring)
        {
            var t = new Texture2D(size, size, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[size * size];
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    float dx = Mathf.Max(0, Mathf.Max(r - x - 0.5f, x + 0.5f - (size - r)));
                    float dy = Mathf.Max(0, Mathf.Max(r - y - 0.5f, y + 0.5f - (size - r)));
                    float d = Mathf.Sqrt(dx * dx + dy * dy) - r;            // <0 inside
                    float a = Mathf.Clamp01(0.5f - d);
                    if (ring) a *= Mathf.Clamp01(d + 2.5f);                 // 2 px outline
                    px[y * size + x] = new Color32(255, 255, 255, (byte)(a * 255));
                }
            t.SetPixels32(px);
            t.Apply();
            return Sprite.Create(t, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f), 100, 0, SpriteMeshType.FullRect, new Vector4(r, r, r, r));
        }

        static Sprite MakeCircle(int size)
        {
            var t = new Texture2D(size, size, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[size * size];
            float c = (size - 1) / 2f, R = size / 2f - 1;
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    float d = Mathf.Sqrt((x - c) * (x - c) + (y - c) * (y - c)) - R;
                    px[y * size + x] = new Color32(255, 255, 255, (byte)(Mathf.Clamp01(0.5f - d) * 255));
                }
            t.SetPixels32(px);
            t.Apply();
            return Sprite.Create(t, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f));
        }

        public static RectTransform Rect(string name, Transform parent)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            return (RectTransform)go.transform;
        }

        public static RectTransform Anchor(this RectTransform r, Vector2 min, Vector2 max, Vector2 pivot, Vector2 pos, Vector2 size)
        {
            r.anchorMin = min; r.anchorMax = max; r.pivot = pivot; r.anchoredPosition = pos; r.sizeDelta = size;
            return r;
        }

        public static RectTransform Fill(this RectTransform r, float pad = 0)
        {
            r.anchorMin = Vector2.zero; r.anchorMax = Vector2.one; r.offsetMin = new Vector2(pad, pad); r.offsetMax = new Vector2(-pad, -pad);
            return r;
        }

        public static Image Panel(string name, Transform parent, Color color, float radiusScale = 1f, bool outline = true, bool frosted = false)
        {
            var r = Rect(name, parent);
            var img = r.gameObject.AddComponent<Image>();
            img.sprite = Round; img.type = Image.Type.Sliced; img.color = color; img.pixelsPerUnitMultiplier = 1f / radiusScale;
            if (frosted && GlassMat != null) img.material = GlassMat;
            if (outline)
            {
                var o = Rect("Edge", r).Fill();
                var oi = o.gameObject.AddComponent<Image>();
                oi.sprite = Ring; oi.type = Image.Type.Sliced; oi.color = Edge; oi.raycastTarget = false; oi.pixelsPerUnitMultiplier = 1f / radiusScale;
                o.gameObject.AddComponent<LayoutElement>().ignoreLayout = true;
            }
            return img;
        }

        public static Text Label(string name, Transform parent, string text, int size, Color color, TextAnchor align = TextAnchor.MiddleLeft, FontStyle style = FontStyle.Normal)
        {
            var r = Rect(name, parent);
            var t = r.gameObject.AddComponent<Text>();
            t.font = Font; t.text = text; t.fontSize = size; t.color = color; t.alignment = align; t.fontStyle = style;
            t.horizontalOverflow = HorizontalWrapMode.Wrap; t.verticalOverflow = VerticalWrapMode.Overflow;
            t.raycastTarget = false; t.lineSpacing = 1.12f; t.supportRichText = true;
            return t;
        }

        public static Image Icon(string name, Transform parent, Texture2D tex, Color color, float size)
        {
            var r = Rect(name, parent);
            r.sizeDelta = new Vector2(size, size);
            var img = r.gameObject.AddComponent<Image>();
            if (tex != null) img.sprite = Sprite.Create(tex, new UnityEngine.Rect(0, 0, tex.width, tex.height), new Vector2(0.5f, 0.5f));
            img.color = color; img.raycastTarget = false; img.preserveAspect = true;
            return img;
        }

        public static Button Pill(string name, Transform parent, string text, UnityAction onClick, int size = 20, Color? fill = null)
        {
            var bg = Panel(name, parent, fill ?? GlassHi, 0.8f);
            var b = bg.gameObject.AddComponent<Button>();
            var cb = b.colors; cb.highlightedColor = new Color(1.25f, 1.25f, 1.25f, 1); cb.pressedColor = new Color(0.8f, 0.9f, 1f, 1); b.colors = cb;
            b.onClick.AddListener(onClick);
            var l = Label("Text", bg.transform, text, size, Ink, TextAnchor.MiddleCenter);
            l.rectTransform.Fill(6);
            var fit = bg.gameObject.AddComponent<LayoutElement>();
            fit.preferredHeight = size * 2.1f;
            fit.preferredWidth = l.preferredWidth + size * 1.6f;
            return b;
        }

        public static Button IconButton(string name, Transform parent, Texture2D icon, UnityAction onClick, float size, Color bgColor)
        {
            var r = Rect(name, parent);
            r.sizeDelta = new Vector2(size, size);
            var img = r.gameObject.AddComponent<Image>();
            img.sprite = Circle; img.color = bgColor;
            var b = r.gameObject.AddComponent<Button>();
            b.onClick.AddListener(onClick);
            var ic = Icon("Icon", r, icon, Ink, size * 0.55f);
            ic.rectTransform.anchoredPosition = Vector2.zero;
            return b;
        }
    }
}
