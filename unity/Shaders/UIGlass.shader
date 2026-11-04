// Frosted glass for uGUI (built-in pipeline): blurs what is behind the panel (GrabPass),
// darkens it slightly, adds a faint white veil and a soft top highlight. Uses the Image sprite alpha
// (rounded rect) as the shape and vertex colour alpha as overall opacity.
Shader "Holo/UIGlass"
{
    Properties
    {
        [PerRendererData] _MainTex ("Sprite", 2D) = "white" {}
        _Blur ("Blur (px)", Range(0, 12)) = 6
        _Darken ("Darken", Range(0, 1)) = 0.32
        _Veil ("White veil", Range(0, 0.5)) = 0.07
        _Highlight ("Top highlight", Range(0, 0.3)) = 0.07
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" "PreviewType"="Plane" "CanUseSpriteAtlas"="True" }
        Cull Off ZWrite Off ZTest [unity_GUIZTestMode]
        Blend SrcAlpha OneMinusSrcAlpha

        GrabPass { "_HoloGlassGrab" }

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            sampler2D _HoloGlassGrab;
            float4 _HoloGlassGrab_TexelSize;
            float _Blur, _Darken, _Veil, _Highlight;

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; float4 color : COLOR; };
            struct v2f { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; float4 grab : TEXCOORD1; float4 color : COLOR; };

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                o.grab = ComputeGrabScreenPos(o.pos);
                o.color = v.color;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float2 uv = i.grab.xy / i.grab.w;
                float2 px = _HoloGlassGrab_TexelSize.xy * _Blur;
                // 13-tap blur (two rings)
                fixed3 c = tex2D(_HoloGlassGrab, uv).rgb * 0.16;
                const float2 d1[6] = { float2(1,0), float2(0.5,0.866), float2(-0.5,0.866), float2(-1,0), float2(-0.5,-0.866), float2(0.5,-0.866) };
                [unroll] for (int k = 0; k < 6; k++)
                {
                    c += tex2D(_HoloGlassGrab, uv + d1[k] * px).rgb * 0.08;
                    c += tex2D(_HoloGlassGrab, uv + d1[k].yx * px * 2.2).rgb * 0.06;
                }
                c = c * (1 - _Darken) + _Veil;
                c += _Highlight * saturate(i.uv.y) * saturate(i.uv.y);
                float a = tex2D(_MainTex, i.uv).a * i.color.a;
                return fixed4(c, a);
            }
            ENDCG
        }
    }
}
