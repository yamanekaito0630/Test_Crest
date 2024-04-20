using System;
using System.Collections;
using System.Net.Mail;
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.PlayerLoop;
using UnityEngine.U2D;
using Random = UnityEngine.Random;

public class PlayerIM_Round_V1 : Agent
{
    // ノード番号
    public int nodeIndex;
    public GameObject DetactionArea_V1;
    
    // 各種パラメータ
    public float speed = 470.0f;
    public float rotSpeed = 7.0f;
    // public float propulsion = 30.0f;
    public float defaultDrag = 0.0f;
    public float waterDrag = 20.0f;
    public Vector3 initPos;
    private Quaternion initRot;

    // 上下左右の位置センサ
    public GameObject sensorLeft;
    public GameObject sensorRight;
    public GameObject sensorTop;
    public GameObject sensorBottom;

    // ファン
    public GameObject rightFrontFan;
    public GameObject leftFrontFan;
    public GameObject rightMiddleFan;
    public GameObject leftMiddleFan;
    public GameObject rightBehindFan;
    public GameObject leftBehindFan;

    // Directional Cameras
    public GameObject camera1;
    public GameObject camera2;
    public GameObject camera3;
    public GameObject camera4;
    public GameObject camera5;
    public GameObject camera6;

    // Boid Area
    public GameObject myBoidIn;
    public GameObject myBoidOut;
    
    // LEDs
    public GameObject frontLED;
    public GameObject backLED;

    private Rigidbody playerRb;
    private Rigidbody rightFrontFanRb;
    private Rigidbody leftFrontFanRb;
    private Rigidbody rightMiddleFanRb;
    private Rigidbody leftMiddleFanRb;
    private Rigidbody rightBehindFanRb;
    private Rigidbody leftBehindFanRb;
    
    // 追加センサ入力値
    // private float directionalValue;
    private float collisionValue;
    private float targetAreaValue;
    
    // private float posCosSim;
    // private float cosSim;

    private Vector3 posGoal1;
    private Vector3 posGoal2;
    private float prevToGoal;
    private float toGoal;
    
    private bool isAboveOcean;

    // private Vector4 idealPosture;
    // private Vector4 actualPosture;

    private int cptCount;
    private float vSim;

    private float distanceReward;
    private float fitness1;
    private float fitness2;
    private float fitness3;
    private float fitness4;
    private float fitness5;
    private float fitness6;
    private float fitness7;

    // ターゲットに近づくため
    private float k1 = 0.0f;
    // 未実装
    private float k2 = 0.0f;
    // ターゲットに到達時
    private float k3 = 1.0f;
    // 衝突時
    private float k4 = 0.0055f;
    // 群れるため
    private float k5 = 0.00000f;
    // 速度類似度
    private float k6 = 0.00f;
    // 姿勢の制御
    private float k7 = 0.00f;

    private float xRot = 0.0f;
    private float yRot = 0.0f;
    private float zRot = 0.0f;


    public override void Initialize()
    {
        playerRb = GetComponent<Rigidbody>();
        
        rightFrontFanRb = rightFrontFan.GetComponent<Rigidbody>();
        leftFrontFanRb = leftFrontFan.GetComponent<Rigidbody>();
        rightMiddleFanRb = rightMiddleFan.GetComponent<Rigidbody>();
        leftMiddleFanRb = leftMiddleFan.GetComponent<Rigidbody>();
        rightBehindFanRb = rightBehindFan.GetComponent<Rigidbody>();
        leftBehindFanRb = leftBehindFan.GetComponent<Rigidbody>();

        posGoal1 = GameObject.FindWithTag("CheckPoint1").transform.position;
        posGoal2 = GameObject.FindWithTag("CheckPoint2").transform.position;
        
        isAboveOcean = true;
        
        // directionalValue = 0.0f;

        // idealPosture = new Vector4(1.0f, 1.0f, -0.15f, -0.15f);
        // actualPosture = new Vector4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    public override void OnEpisodeBegin()
    {
        // SceneManager.LoadScene(0);

        playerRb.velocity = Vector3.zero;
        initRot = Quaternion.Euler(0.0f, 0.0f, 0.0f);
        
        // 初期化処理
        transform.position = initPos;
        transform.rotation = initRot;
        
        toGoal = Vector3.Distance(playerRb.transform.position, posGoal1);
        prevToGoal = toGoal;
        
        // posCosSim = 0.0f;
        // cosSim = 0.0f;

        cptCount = 0;
        vSim = 0.0f;

        collisionValue = 0.0f;
        targetAreaValue = 0.0f;

        distanceReward = 0.0f;
        fitness1 = 0.0f;
        fitness2 = 0.0f;
        fitness3 = 0.0f;
        fitness4 = 0.0f;
        fitness5 = 0.0f;
        fitness6 = 0.0f;
        fitness7 = 0.0f;

        // エピソード開始時に水中ドローンに初期推進力を与える
        // playerRb.AddForce(transform.TransformDirection(new Vector3(0, 200.0f, 200.0f)));
        xRot = 0.0f;
        yRot = 0.0f;
        zRot = 0.0f;
    }

    public void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("CheckPoint1"))
        {
            if (cptCount % 2 == 0)
            {
                AddReward(1.0f * k3);
                fitness3 += 1.0f * k3;
                cptCount++;
                
                prevToGoal = Vector3.Distance(playerRb.transform.position, posGoal2);

                // EndEpisode();
            }
        }
        
        if (other.CompareTag("CheckPoint2"))
        {
            if (cptCount % 2 == 1)
            {
                AddReward(1.0f * k3);
                fitness3 += 1.0f * k3;
                cptCount++;
                
                prevToGoal = Vector3.Distance(playerRb.transform.position, posGoal1);
            }
        }
    }
    
    public void OnTriggerStay(Collider other)
    {
        if (other.CompareTag("CheckPoint1"))
        {
            if (cptCount % 2 == 1)
            {
                targetAreaValue = 1.0f;
                // Debug.Log("check1");
            }
            // EndEpisode();
        }
        
        if (other.CompareTag("CheckPoint2"))
        {
            if (cptCount % 2 == 0)
            {
                targetAreaValue = 1.0f;
                // Debug.Log("check2");
            }
            // EndEpisode();
        }

        //  危険区域を設けた場合
        // if (other.CompareTag("DangerArea"))
        // {
        //     // 衝突ペナルティの計算
        //     AddReward(-1.0f * k4);
        //     fitness4 += -1.0f * k4;
        //     collisionValue = 1.0f;
        // }

        if (other.CompareTag("BoidOut"))
        {
            if (other.gameObject != myBoidOut)
            {
                AddReward(1.0f * k5);
                fitness5 += 1.0f * k5;
                Debug.Log("boid!!");

                // Debug.Log(other.transform.parent.GetComponent<Rigidbody>().velocity);
                vSim = CosineSimilarity(other.transform.parent.GetComponent<Rigidbody>().transform.forward, playerRb.transform.forward);
                if (!Double.IsNaN(vSim))
                {
                    AddReward(vSim * k6);
                    fitness6 += vSim * k6;
                     Debug.Log(vSim);
                }
            }
        }
    }
    
    public void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("CheckPoint1") || other.CompareTag("CheckPoint2"))
        {
            targetAreaValue = 0.0f;
        }

        if (other.CompareTag("DangerArea"))
        {
            collisionValue = 0.0f;
        }
    }
    
    public void OnCollisionStay(Collision other)
    {
        // タグが"Obstacle"のオブジェクトに衝突している場合
        if (other.gameObject.CompareTag("Obstacle") || other.gameObject.CompareTag("Robot"))
        {
            // 衝突ペナルティの計算
            AddReward(-1.0f * k4);
            fitness4 += -1.0f * k4;
            collisionValue = 1.0f;
            // EndEpisode();
        }
    }
    
    public void OnCollisionExit(Collision other)
    {
        if (other.gameObject.CompareTag("Obstacle"))
        {
            collisionValue = 0.0f;
        }
    }

    public void AboveOcean()
    {
        // 水上の処理
        playerRb.drag = defaultDrag;
        isAboveOcean = true;
    }

    public void BelowOcean()
    {
        // 水中の処理
        playerRb.drag = waterDrag;
        isAboveOcean = false;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 現在の各報酬を取得
        sensor.AddObservation(fitness1);
        sensor.AddObservation(fitness2);
        sensor.AddObservation(fitness3);
        sensor.AddObservation(fitness4);
        sensor.AddObservation(fitness5);
        sensor.AddObservation(fitness6);
        sensor.AddObservation(fitness7);

        // 速度と位置
        sensor.AddObservation(playerRb.velocity.magnitude);
        sensor.AddObservation(playerRb.transform.position);
        
        // 他ロボットとのリンク
        sensor.AddObservation(DetactionArea_V1.GetComponent<DetactionArea_V1>().firstNeighborhood);
        sensor.AddObservation(DetactionArea_V1.GetComponent<DetactionArea_V1>().secondNeighborhood);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (nodeIndex == 0)
        {
            Debug.Log("第1近傍："+DetactionArea_V1.GetComponent<DetactionArea_V1>().firstNeighborhood);
            Debug.Log("第2近傍："+DetactionArea_V1.GetComponent<DetactionArea_V1>().secondNeighborhood);
        }


        // アクションを取得
        float horizontalInput = Mathf.Abs(actions.ContinuousActions[0]);
        float verticalInput = actions.ContinuousActions[1];
        float rotYInput = actions.ContinuousActions[2];
        // float rotXInput = actions.ContinuousActions[3];
        // float rotZInput = actions.ContinuousActions[4];

        // float redInput = actions.ContinuousActions[3];
        // float blueInput = actions.ContinuousActions[4];

        // float rightFrontInput = actions.ContinuousActions[0];
        // float leftFrontInput = actions.ContinuousActions[1];
        // float rightMiddleInput = actions.ContinuousActions[2];
        // float leftMiddleInput = actions.ContinuousActions[3];
        // float rightBehindInput = actions.ContinuousActions[4];
        // float leftBehindInput = actions.ContinuousActions[5];

        // デバッグ用アクション
        // rightBehindInput = 1.0f;
        // leftBehindInput = 1.0f;
        // rightFrontInput = -0.15f;
        // leftFrontInput = -0.15f;
        
        // LEDの点灯
        // if (redInput > 0.0f)
        // {
        //     frontLED.GetComponent<Renderer>().material.color = Color.red;
        // }
        // else
        // {
        //     frontLED.GetComponent<Renderer>().material.color = Color.gray;
        // }
        //
        // if (blueInput > 0.0f)
        // {
        //     backLED.GetComponent<Renderer>().material.color = Color.blue;
        // }
        // else
        // {
        //     backLED.GetComponent<Renderer>().material.color = Color.gray;
        // }
        

        if (isAboveOcean)
        {
            verticalInput = 0.0f;
            horizontalInput = 0.0f;
            rotYInput = 0.0f;
            // rotXInput = 0.0f;
            // rotZInput = 0.0f;

            // rightFrontInput = 0.0f;
            // leftFrontInput = 0.0f;
            // rightMiddleInput = 0.0f;
            // leftMiddleInput = 0.0f;
            // rightBehindInput = 0.0f;
            // leftBehindInput = 0.0f;
        }

        playerRb.AddForce(transform.forward * horizontalInput * speed);
        playerRb.AddForce(transform.up * verticalInput * speed);
        
        Vector3 localAngle = this.transform.localRotation.eulerAngles;
        localAngle.y = yRot;
        transform.localEulerAngles = localAngle;
        yRot += rotYInput * rotSpeed;
        // transform.rotation = Quaternion.AngleAxis(rotYInput * rotSpeed, Vector3.up) * transform.rotation;

        float postureReward = (1.0f / (Mathf.Abs(playerRb.transform.rotation.eulerAngles.x) + 1.0f)) + (1.0f / (playerRb.transform.rotation.eulerAngles.z + 1.0f));
        AddReward(postureReward * k7);
        fitness7 += postureReward * k7;

        // 各ファンにForceを適用
        // rightFrontFanRb.AddRelativeForce(new Vector3(0.0f, rightFrontInput * propulsion, 0.0f));
        // leftFrontFanRb.AddRelativeForce(new Vector3(0.0f, leftFrontInput * propulsion, 0.0f));
        // rightMiddleFanRb.AddRelativeForce(new Vector3(0.0f, rightMiddleInput * propulsion, 0.0f));
        // leftMiddleFanRb.AddRelativeForce(new Vector3(0.0f, leftMiddleInput * propulsion, 0.0f));
        // rightBehindFanRb.AddRelativeForce(new Vector3(0.0f, rightBehindInput * propulsion, 0.0f));
        // leftBehindFanRb.AddRelativeForce(new Vector3(0.0f, leftBehindInput * propulsion, 0.0f));

        // カメラセンサが他ロボットのLEDを検出した場合の処理
        if (camera1.GetComponent<DirectionalCamera>().isDetactRed || camera1.GetComponent<DirectionalCamera>().isDetactBlue)
        {
            AddReward(1.0f * k2);
            fitness2 += 1.0f * k2;
            // Debug.Log("detact!!");
        }

        // コサイン類似度の報酬の計算
        // actualPosture = new Vector4(rightBehindInput, leftBehindInput, rightFrontInput, leftFrontInput);
        // posCosSim = PostureCosineSimilarity(idealPosture, actualPosture);
        // cosSim = CosineSimilarity(playerRb.transform.forward, playerRb.velocity);
        
        // 距離報酬の計算
        if (cptCount % 2 == 0)
        {
            toGoal = Vector3.Distance(playerRb.transform.position, posGoal1);
        }
        if (cptCount % 2 == 1)
        {
            toGoal = Vector3.Distance(playerRb.transform.position, posGoal2);
        }
		
        distanceReward = prevToGoal - toGoal;
        //distanceReward = CulculateDistanceReward(toGoal);
        if (!Double.IsNaN(distanceReward))
        {
            AddReward(distanceReward * k1);
            fitness1 += distanceReward * k1;
        }

        prevToGoal = toGoal;
        // Debug.Log(fitness1);
    }

    private float CosineSimilarity(Vector3 d, Vector3 v)
    {
        return Vector3.Dot(d, v) / (d.magnitude * v.magnitude);
    }

    private float PostureCosineSimilarity(Vector4 i, Vector4 a)
    {
        return Vector4.Dot(i, a) / (i.magnitude * a.magnitude);
    }

    private float CulculateDistanceReward(float toGoal)
    {
        float c1 = 1.0f;
        float c2 = 0.0f;
        float alpha = 0.5f;

        float res = (1.0f / (c1 + Mathf.Abs(Mathf.Pow(toGoal, alpha)))) - c2;
        if (res <= 0.0f)
        {
            res = 0.0f;
        }

        return res;
    }

    // 手動テスト用
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // デフォルトの入力
        float horizontalInput = Input.GetAxis("Vertical");
        float verticalInput = Input.GetKey(KeyCode.Space) ? 1.0f : 0.0f;
        float rotYInput = Input.GetAxis("Horizontal");
        // float rotXInput = Input.GetAxis("Horizontal");
        // float rotZInput = 0.0f;
        
        // float redInput = 1.0f;
        // float blueInput = 1.0f;
        
        // float leftFrontInput = Input.GetKey(KeyCode.Q) ? 1.0f : 0.0f;
        // float leftMiddleInput = Input.GetKey(KeyCode.A) ? 1.0f : 0.0f;
        // float leftBehindInput = Input.GetKey(KeyCode.Z) ? 1.0f : 0.0f;
        // float rightFrontInput = Input.GetKey(KeyCode.W) ? 1.0f : 0.0f;
        // float rightMiddleInput = Input.GetKey(KeyCode.S) ? 1.0f : 0.0f;
        // float rightBehindInput = Input.GetKey(KeyCode.X) ? 1.0f : 0.0f;

        // 入力をエージェントのアクションに割り当て
        ActionSegment<float> continuousAct = actionsOut.ContinuousActions;
        continuousAct[0] = horizontalInput;
        continuousAct[1] = verticalInput;
        continuousAct[2] = rotYInput;
        
        // continuousAct[3] = rotXInput;
        // continuousAct[4] = rotZInput;

        // continuousAct[3] = redInput;
        // continuousAct[4] = blueInput;

        // continuousAct[0] = rightFrontInput;
        // continuousAct[1] = leftFrontInput;
        // continuousAct[2] = rightMiddleInput;
        // continuousAct[3] = leftMiddleInput;
        // continuousAct[4] = rightBehindInput;
        // continuousAct[5] = leftBehindInput;
    }
}
