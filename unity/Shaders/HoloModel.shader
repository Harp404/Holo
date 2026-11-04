// Hologram look for generated 3D models: translucent blue body, bright fresnel rim, moving scanlines.
Shader "Holo/HoloModel"
{
    Properties
    {
        _Color ("Body", Color) = (0.35, 0.65, 1.0, 0.18)
        _Rim ("Rim", Color) = (0.65, 0.88, 1.0, 1.0)
        _RimPower ("Rim power", Range(0.5, 6)) = 2.2
        _Scan ("Scanline strength", Range(0, 1)) = 0.25
        _ScanDensity ("Scanline density", Float) = 90
        _Alpha ("Alpha", Range(0, 1)) = 1
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        // depth prepass so the hologram doesn't show its own back faces through itself
        Pass { ZWrite On ColorMask 0 }
        Pass
        {
            ZWrite Off
            Blend SrcAlpha One
            Cull Back
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            fixed4 _Color, _Rim;
            float _RimPower, _Scan, _ScanDensity, _Alpha;
            struct v2f { float4 pos : SV_POSITION; float3 n : TEXCOORD0; float3 v : TEXCOORD1; float3 w : TEXCOORD2; };
            v2f vert (appdata_base v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.n = UnityObjectToWorldNormal(v.normal);
                o.w = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.v = normalize(_WorldSpaceCameraPos - o.w);
                return o;
            }
            fixed4 frag (v2f i) : SV_Target
            {
                float f = pow(1 - saturate(abs(dot(normalize(i.n), i.v))), _RimPower);
                float scan = 0.5 + 0.5 * sin(i.w.y * _ScanDensity - _Time.y * 3.0);
                float3 c = _Color.rgb * (0.6 + 0.4 * scan * _Scan * 4) + _Rim.rgb * f * 1.4;
                float a = saturate(_Color.a + f * 0.85 + scan * _Scan * 0.08) * _Alpha;
                return fixed4(c, a);
            }
            ENDCG
        }
    }
}
