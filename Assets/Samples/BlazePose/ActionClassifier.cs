using System.Threading;
using Cysharp.Threading.Tasks;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Windows;
using System;

namespace TensorFlowLite
{
    public sealed class ActionClassifier : BaseActionPredictor<float>
    {
        public class Result
        {
            public float[] classificationResult;
            public string classificationResultLabel;
        }

        private readonly float[] output0 = new float[12];

        private readonly Result result;


        public ActionClassifier(string modelPath) : base(modelPath, true)
        {

            result = new Result()
            {
                classificationResult = new float[12],
                classificationResultLabel = "None"
            };
        }

        public void Invoke(float[] inputArray)
        {
            interpreter.SetInputTensorData(0, inputArray);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
        }

        public override void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture, PalmDetect.Result)");
        }

        public async UniTask<Result> InvokeAsync(float[] inputArray, CancellationToken cancellationToken, PlayerLoopTiming timing)
        {
            await UniTask.SwitchToThreadPool();

            interpreter.SetInputTensorData(0, inputArray);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);

            var results = GetResult();

            await UniTask.SwitchToMainThread(timing, cancellationToken);
            return results;
        }

        public Result GetResult()
        {
            string[] label = new string[]{"squat_down", "neutral_stand_front", "neutral_stand_side", "pushup_up", "pushup_down", "bicycle_left", "bicycle_right", "highKnee_left", "highKnee_right", "jumpingJack_out", "sitUp_down", "sitUp_up"};

            float MaxValue = 0;
            int MaxIndex = 0;
            for (int i = 0; i < output0.Length; ++i)
            {
                if (output0[i] > MaxValue)
                {
                    MaxValue = output0[i];
                    MaxIndex = i;
                }
            }

            result.classificationResultLabel = label[MaxIndex];

            for (int i = 0; i < output0.Length; ++i) 
            {
                result.classificationResult[i] = output0[i];
            }

            return result;
        }

        public float[] ConvertToEmbedding(Vector4[] viewportLandmarks)
        {
            /*
            .midpointDistance(.leftHip, .rightHip, .leftShoulder, .rightShoulder),
            .distance(.leftShoulder, .leftElbow),
            .distance(.rightShoulder, .rightElbow),
            .distance(.leftElbow, .leftWrist),
            .distance(.rightElbow, .rightWrist),
            .distance(.leftHip, .leftKnee),
            .distance(.rightHip, .rightKnee),
            .distance(.leftKnee, .leftAnkle),
            .distance(.rightKnee, .rightAnkle),

            // Two joints.
            .distance(.leftShoulder, .leftWrist),
            .distance(.rightShoulder, .rightWrist),
            .distance(.leftHip, .leftAnkle),
            .distance(.rightHip, .rightAnkle),

            // Four joints.
            .distance(.leftHip, .leftWrist),
            .distance(.rightHip, .rightWrist),

            // Five joints.
            .distance(.leftShoulder, .leftAnkle),
            .distance(.rightShoulder, .rightAnkle),
            .distance(.leftHip, .leftWrist),
            .distance(.rightHip, .rightWrist),

            // Cross body.
            .distance(.leftElbow, .rightElbow),
            .distance(.leftKnee, .rightKnee),
            .distance(.leftWrist, .rightWrist),
            .distance(.leftAnkle, .rightAnkle),
            */


            // viewportLandmarks[23], viewportLandmarks[24], viewportLandmarks[11], viewportLandmarks[12]
            // 11, 13
            // 12, 14
            // 13, 15
            // 14, 16
            // 23, 25
            // 24, 26
            // 25, 27
            // 26, 28

            // 11, 15
            // 12, 16
            // 23, 27
            // 24, 28

            // 23, 15
            // 24, 16

            // 11, 27
            // 12, 28
            // 23, 15
            // 24, 16

            // 13, 14
            // 25, 26
            // 15, 16
            // 27, 28

            float[] embeddingFlat = new float[69];

            float[][] embedding = new float[23][];

            embedding[0] = MidPointDistance(viewportLandmarks[23], viewportLandmarks[24], viewportLandmarks[11], viewportLandmarks[12]);

            embedding[1] = Distance(viewportLandmarks[11], viewportLandmarks[13]);
            embedding[2] = Distance(viewportLandmarks[12], viewportLandmarks[14]);
            embedding[3] = Distance(viewportLandmarks[13], viewportLandmarks[15]);
            embedding[4] = Distance(viewportLandmarks[14], viewportLandmarks[16]);
            embedding[5] = Distance(viewportLandmarks[23], viewportLandmarks[25]);
            embedding[6] = Distance(viewportLandmarks[24], viewportLandmarks[26]);
            embedding[7] = Distance(viewportLandmarks[25], viewportLandmarks[27]);
            embedding[8] = Distance(viewportLandmarks[26], viewportLandmarks[28]);

            embedding[9] = Distance(viewportLandmarks[11], viewportLandmarks[15]);
            embedding[10] = Distance(viewportLandmarks[12], viewportLandmarks[16]);
            embedding[11] = Distance(viewportLandmarks[23], viewportLandmarks[27]);
            embedding[12] = Distance(viewportLandmarks[24], viewportLandmarks[28]);

            embedding[13] = Distance(viewportLandmarks[23], viewportLandmarks[15]);
            embedding[14] = Distance(viewportLandmarks[24], viewportLandmarks[16]);

            embedding[15] = Distance(viewportLandmarks[11], viewportLandmarks[27]);
            embedding[16] = Distance(viewportLandmarks[12], viewportLandmarks[28]);
            embedding[17] = Distance(viewportLandmarks[23], viewportLandmarks[15]);
            embedding[18] = Distance(viewportLandmarks[24], viewportLandmarks[16]);

            embedding[19] = Distance(viewportLandmarks[13], viewportLandmarks[14]);
            embedding[20] = Distance(viewportLandmarks[25], viewportLandmarks[26]);
            embedding[21] = Distance(viewportLandmarks[15], viewportLandmarks[16]);
            embedding[22] = Distance(viewportLandmarks[27], viewportLandmarks[28]);


            // Flatten
            for (int i = 0; i < 69 / 3; ++i)
            {
                embeddingFlat[i * 3 + 0] = embedding[i][0];
                embeddingFlat[i * 3 + 1] = embedding[i][1];
                embeddingFlat[i * 3 + 2] = embedding[i][2];
            }

            return embeddingFlat;
        }

        private float[] Distance(Vector4 firstLandmark, Vector4 secondLandmark)
        {
            float[] distance = new float[3];
            distance[0] = secondLandmark.x - firstLandmark.x;
            distance[1] = secondLandmark.y - firstLandmark.y;
            distance[2] = secondLandmark.z - firstLandmark.z;
            return distance;
        }

        private float[] MidPointDistance(Vector4 firstLandmark, Vector4 secondLandmark, Vector4 thirdLandmark, Vector4 fourthLandmark)
        {
            float[] distance = new float[3];
            distance[0] = (thirdLandmark.x + fourthLandmark.x) * 0.5f - (firstLandmark.x + secondLandmark.x) * 0.5f;
            distance[1] = (thirdLandmark.y + fourthLandmark.y) * 0.5f - (firstLandmark.y + secondLandmark.y) * 0.5f;
            distance[2] = (thirdLandmark.z + fourthLandmark.z) * 0.5f - (firstLandmark.z + secondLandmark.z) * 0.5f;
            return distance;
        }

        public Vector4[] NormalizePoseLandmarks(Vector4[] viewportLandmarks, float width, float height)
        {
            Vector4[] outputViewportLandmarks = new Vector4[viewportLandmarks.Length];
            Vector4[] temViewportLandmarks = new Vector4[viewportLandmarks.Length];

            for (int i = 0; i < viewportLandmarks.Length; ++i)
            {
                temViewportLandmarks[i].x = viewportLandmarks[i].x * 1;
                temViewportLandmarks[i].y = viewportLandmarks[i].y * 1;
                temViewportLandmarks[i].z = viewportLandmarks[i].z * 0;
                temViewportLandmarks[i].w = viewportLandmarks[i].w;
            }

            Vector4 pose_center = GetPoseCenter(temViewportLandmarks);

            for (int i = 0; i < temViewportLandmarks.Length; ++i)
            {
                outputViewportLandmarks[i] = temViewportLandmarks[i] - pose_center;
            }

            float torso_size = GetTorsoSizeMultiplier(outputViewportLandmarks);
            for (int i = 0; i < outputViewportLandmarks.Length; ++i)
            {
                outputViewportLandmarks[i] = outputViewportLandmarks[i] / torso_size;
            }
            // throw new Exception("HI: " + torso_size.ToString());


            for (int i = 0; i < outputViewportLandmarks.Length; ++i)
            {
                outputViewportLandmarks[i] = outputViewportLandmarks[i] * 100;
            }
            return outputViewportLandmarks;
        }

        private Vector4 GetPoseCenter(Vector4[] viewportLandmarks)
        {
            //Hip center
            Vector4 poseCenter = new Vector4(0, 0, 0, 0);
            poseCenter.x = (viewportLandmarks[23].x + viewportLandmarks[24].x) * 0.5f;
            poseCenter.y = (viewportLandmarks[23].y + viewportLandmarks[24].y) * 0.5f;
            poseCenter.z = (viewportLandmarks[23].z + viewportLandmarks[24].z) * 0.5f;
            poseCenter.w = (viewportLandmarks[23].w + viewportLandmarks[24].w) * 0.5f;
            return poseCenter;
        }

        private float GetTorsoSizeMultiplier(Vector4[] viewportLandmarks)
        {
            float torso_size_multiplier = 2.5f;

            // Hip center
            float[] hipCenter = new float[2];
            hipCenter[0] = (viewportLandmarks[23].x + viewportLandmarks[24].x) * 0.5f;
            hipCenter[1] = (viewportLandmarks[23].y + viewportLandmarks[24].y) * 0.5f;

            // Shoulder center
            float[] shoulderCenter = new float[2];
            shoulderCenter[0] = (viewportLandmarks[11].x + viewportLandmarks[12].x) * 0.5f;
            shoulderCenter[1] = (viewportLandmarks[11].y + viewportLandmarks[12].y) * 0.5f;

            // Torso size as the minimum body size.
            float torsoSize = (float)Math.Pow(Math.Pow(shoulderCenter[0] - hipCenter[0], 2) + Math.Pow(shoulderCenter[1] - hipCenter[1], 2), 0.5);
        
            // Max dist to pose center.
            float[] normViewLandmarks = new float[viewportLandmarks.Length];
            float maxDist = 0;
            for (int i = 0; i < normViewLandmarks.Length; ++i)
            {
                normViewLandmarks[i] = (float)Math.Pow(Math.Pow(viewportLandmarks[i][0] - hipCenter[0], 2) + Math.Pow(viewportLandmarks[i][1] - hipCenter[1], 2), 0.5);
                if (normViewLandmarks[i] > maxDist)
                {
                    maxDist = normViewLandmarks[i];
                }
            }
            return (float)Math.Max(torsoSize * torso_size_multiplier, maxDist);
        }
    }
}


