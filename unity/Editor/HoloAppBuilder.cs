using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Holo.EditorTools
{
    /// <summary>Holo > Build App Scene: camera, passthrough stand-in, Muse voice + panel. Saves Assets/Scenes/HoloApp.unity.</summary>
    public static class HoloAppBuilder
    {
        const string Icons = "Assets/Holo/UI/Icons/";

        static Texture2D Ic(string n) => AssetDatabase.LoadAssetAtPath<Texture2D>(Icons + n + ".png");

        [MenuItem("Holo/Build App Scene")]
        public static string Build()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.03f, 0.05f, 0.09f);
            cam.fieldOfView = 50;
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane = 100;
            camGo.AddComponent<AudioListener>();

            var app = new GameObject("Holo");
            var api = app.AddComponent<HoloApi>();
            var video = app.AddComponent<PassthroughVideo>();
            video.framesFolder = "Passthrough/drone";     // raw camera: every highlight is live (SAM 3.1), nothing baked in
            video.rawFolder = "Passthrough/drone";
            video.fps = 12;
            var wake = app.AddComponent<WakeListener>();
            wake.api = api;
            var voice = app.AddComponent<AudioSource>();
            voice.playOnAwake = false;
            var viewer = app.AddComponent<ModelViewer>();
            viewer.head = cam;
            var holo = app.AddComponent<HoloApp>();
            holo.viewer = viewer;
            holo.cam = cam; holo.video = video; holo.api = api; holo.wake = wake; holo.voice = voice;
            holo.font = AssetDatabase.LoadAssetAtPath<Font>("Assets/Holo/UI/Fonts/Inter.ttf");
            holo.icMic = Ic("mic"); holo.icSend = Ic("send"); holo.icSearch = Ic("search"); holo.icClose = Ic("close");
            holo.icMenu = Ic("menu"); holo.icCheck = Ic("check_circle"); holo.icArrow = Ic("arrow_forward");
            holo.icBulb = Ic("lightbulb_outline"); holo.icHand = Ic("pan_tool");

            System.IO.Directory.CreateDirectory("Assets/Scenes");
            const string path = "Assets/Scenes/HoloApp.unity";
            EditorSceneManager.SaveScene(scene, path);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(path, true) };
            return "built " + path;
        }
    }
}
