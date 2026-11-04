// Draws SAM 3 masks over the passthrough view: per-object colour fill + glowing animated edge.
// _Labels: R8 label map (0 = background, k = object k). _Palette: 256x1 RGBA, colour/alpha per object.
// _Target: object index to emphasise (pulse + brighter edge).
Shader "Holo/MaskOverlay"
{
    Properties
    {
        [PerRendererData] _MainTex ("Labels", 2D) = "black" {}
        _Palette ("Palette", 2D) = "black" {}
        _Target ("Target index", Float) = -1
        _EdgePx ("Edge width (px)", Float) = 2.5
        _TexelSize ("Label texel size", Vector) = (0.0015625, 0.0027778, 0, 0)
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" "PreviewType"="Plane" "CanUseSpriteAtlas"="True" }
        Cull Off ZWrite Off ZTest [unity_GUIZTestMode]
        Blend One OneMinusSrcAlpha

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            sampler2D _Palette;
            float _Target, _EdgePx;
            float4 _TexelSize;

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; float4 color : COLOR; };
            struct v2f { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; float4 color : COLOR; };

            v2f vert (appdata v) { v2f o; o.pos = UnityObjectToClipPos(v.vertex); o.uv = v.uv; o.color = v.color; return o; }

            float id(float2 uv) { return round(tex2D(_MainTex, uv).r * 255.0); }

            fixed4 frag (v2f i) : SV_Target
            {
                float c = id(i.uv);
                // edge: any neighbour with a different label
                float2 d = _TexelSize.xy * _EdgePx;
                float n1 = id(i.uv + float2(d.x, 0)), n2 = id(i.uv - float2(d.x, 0));
                float n3 = id(i.uv + float2(0, d.y)), n4 = id(i.uv - float2(0, d.y));
                float edge = (n1 != c || n2 != c || n3 != c || n4 != c) ? 1.0 : 0.0;
                // when outside a mask but next to one, colour the edge with the neighbour's id
                float e = c;
                if (c == 0) e = max(max(n1, n2), max(n3, n4));
                if (e == 0) return 0;

                float4 pal = tex2D(_Palette, float2((e + 0.5) / 256.0, 0.5));
                bool isTarget = abs(e - _Target) < 0.5;
                float pulse = isTarget ? 0.55 + 0.45 * sin(_Time.y * 4.0) : 0.0;
                // moving shimmer along the target edge
                float shimmer = isTarget ? 0.5 + 0.5 * sin((i.uv.x + i.uv.y) * 60.0 - _Time.y * 6.0) : 0.0;

                float fillA = (c != 0) ? pal.a * (isTarget ? (0.28 + 0.18 * pulse) : 1.0) : 0.0;
                float edgeA = edge * (isTarget ? (0.75 + 0.25 * shimmer) : pal.a * 2.2);
                float a = saturate(max(fillA, edgeA)) * i.color.a;
                float3 rgb = pal.rgb * (1.0 + edge * (isTarget ? 0.6 : 0.2));
                return float4(rgb * a, a);
            }
            ENDCG
        }
    }
}
