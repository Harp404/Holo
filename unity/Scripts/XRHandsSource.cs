using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands;

namespace Holo
{
    /// <summary>Real hand tracking through Unity XR Hands (Quest / Meta XR Simulator).</summary>
    public class XRHandsSource : HandSource
    {
        public Transform trackingSpace;   // the XR Origin transform
        public bool rightHand = true;

        XRHandSubsystem _sub;
        static readonly List<XRHandSubsystem> Subs = new();

        public override HandState Get()
        {
            var s = new HandState();
            if (_sub == null || !_sub.running)
            {
                SubsystemManager.GetSubsystems(Subs);
                _sub = Subs.Count > 0 ? Subs[0] : null;
                if (_sub == null) return s;
            }
            var hand = rightHand ? _sub.rightHand : _sub.leftHand;
            if (!hand.isTracked) return s;
            if (!TryJoint(hand, XRHandJointID.IndexTip, out s.indexTip)) return s;
            TryJoint(hand, XRHandJointID.ThumbTip, out s.thumbTip);
            TryJoint(hand, XRHandJointID.Wrist, out s.wrist);
            s.valid = true;
            return s;
        }

        bool TryJoint(XRHand hand, XRHandJointID id, out Vector3 p)
        {
            p = default;
            if (!hand.GetJoint(id).TryGetPose(out var pose)) return false;
            p = trackingSpace != null ? trackingSpace.TransformPoint(pose.position) : pose.position;
            return true;
        }
    }
}
