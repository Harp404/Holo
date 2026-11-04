using UnityEngine;

namespace Holo
{
    public struct HandState
    {
        public bool valid;
        public Vector3 indexTip;
        public Vector3 thumbTip;
        public Vector3 wrist;
    }

    public abstract class HandSource : MonoBehaviour
    {
        public abstract HandState Get();
    }
}
