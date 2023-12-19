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

public class Player : Agent
{
    // 各種パラメータ
    public float speed = 30.0f;
    public float rotSpeed = 1.0f;
    // public float propulsion = 30.0f;
    public float defaultDrag = 0.0f;
    public float waterDrag = 10.0f;
    public Vector3 initPos;
    private Quaternion initRot;

    private GameObject centerPoint;
    private Vector2 centerPos_xz;
    public GameObject center;


    // Target Area
    private Vector3 posGoal1;
    private Vector3 posGoal2;
    private Vector2 posGoal1_xz;
    private Vector2 posGoal2_xz;

    private Rigidbody playerRb;

    private float prevToGoal;
    private float toGoal;
    
    private bool isAboveOcean;

    private GameObject[] robots;
    private int robotNum;

    // private Vector4 idealPosture;
    // private Vector4 actualPosture;

    private int cptCount;

    private int centerCptCount;
    private int prevCptCount;

    private float distanceReward;
    private float fitness1;
    private float fitness2;
    private float fitness3;
    private float fitness4;
    private float fitness5;
    private float fitness6;


    // 中心がターゲットに近づくための係数
    private float k1 = 0.084f;
    // LED（未実装）
    private float k2 = 0.0f;
    // ロボットがターゲットに到達した際の係数
    private float k3 = 1.0f;
    // 衝突時の係数
    private float k4 = 0.005f;
    // 群れるための係数
    private float k5 = 0.00056f;
    // 中心がターゲットに到達した際の係数
    private float k6 = 3.0f;

    public override void Initialize()
    {
        playerRb = GetComponent<Rigidbody>();

        posGoal1 = GameObject.FindWithTag("CheckPoint1").transform.position;
        posGoal2 = GameObject.FindWithTag("CheckPoint2").transform.position;

        posGoal1_xz = new Vector2(posGoal1.x, posGoal1.z);
        posGoal2_xz = new Vector2(posGoal2.x, posGoal2.z);
        
        isAboveOcean = true;

        centerPoint = GameObject.FindWithTag("CenterPoint");

        robots = GameObject.FindGameObjectsWithTag("Robot");
        robotNum = robots.Length;
    }

    public override void OnEpisodeBegin()
    {
        center.GetComponent<Center>().Initialize();

        // SceneManager.LoadScene(0);

        playerRb.velocity = Vector3.zero;
        initRot = Quaternion.Euler(0.0f, 0.0f, 0.0f);
        
        // 初期化処理
        transform.position = initPos;
        transform.rotation = initRot;
        
        centerPos_xz = new Vector2(centerPoint.transform.position.x, centerPoint.transform.position.z);
        toGoal = Vector2.Distance(centerPos_xz, posGoal1_xz);
        prevToGoal = toGoal;
        
        // posCosSim = 0.0f;
        // cosSim = 0.0f;

        cptCount = 0;

        centerCptCount = 0;
        prevCptCount = 0;

        distanceReward = 0.0f;
        fitness1 = 0.0f;
        fitness2 = 0.0f;
        fitness3 = 0.0f;
        fitness4 = 0.0f;
        fitness5 = 0.0f;
        fitness6 = 0.0f;

        // エピソード開始時に水中ドローンに初期推進力を与える
        // playerRb.AddForce(transform.TransformDirection(new Vector3(0, 200.0f, 200.0f)));
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
        if (other.CompareTag("DangerArea"))
        {
            // 衝突ペナルティの計算
            AddReward(-1.0f * k4);
            fitness4 += -1.0f * k4;
        }

        //  群れるための報酬
        if (other.CompareTag("BoidIn"))
        {
            AddReward(-1.0f * k5 / robotNum);
            fitness5 += (-1.0f * k5 / robotNum);
        }

        if (other.CompareTag("BoidOut"))
        {
            AddReward(1.0f * k5 / robotNum);
            fitness5 += 1.0f * k5 / robotNum;
        }
    }
    
    public void OnCollisionStay(Collision other)
    {
        // タグが"Obstacle"のオブジェクトに衝突している場合
        if (other.gameObject.CompareTag("Obstacle") || other.gameObject.CompareTag("Robot"))
        {
            // 衝突ペナルティの計算
            AddReward(-1.0f * k4);
            fitness4 += (-1.0f * k4);

            // EndEpisode();
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

        sensor.AddObservation(playerRb.velocity.magnitude);
        sensor.AddObservation(playerRb.transform.position);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // アクションを取得
        float horizontalInput = Mathf.Abs(actions.ContinuousActions[0]);
        float verticalInput = actions.ContinuousActions[1];
        float rotInput = actions.ContinuousActions[2];

        if (isAboveOcean)
        {
            verticalInput = 0.0f;
            horizontalInput = 0.0f;
            rotInput = 0.0f;
        }

        playerRb.AddForce(transform.forward * horizontalInput * speed);
        playerRb.AddForce(transform.up * verticalInput * speed);
        transform.rotation = Quaternion.AngleAxis(rotInput * rotSpeed, Vector3.up) * transform.rotation;
        
        // 距離報酬の計算
        centerPos_xz = new Vector2(centerPoint.transform.position.x, centerPoint.transform.position.z);
        if (centerCptCount % 2 == 0)
        {
            toGoal = Vector2.Distance(centerPos_xz, posGoal1_xz);
        }
        if (centerCptCount % 2 == 1)
        {
            toGoal = Vector2.Distance(centerPos_xz, posGoal2_xz);
        }
		
        distanceReward = prevToGoal - toGoal;
        // distanceReward = CulculateDistanceReward(toGoal);
        if (!Double.IsNaN(distanceReward))
        {
            AddReward(distanceReward * k1);
            fitness1 += distanceReward * k1;
        }

        prevToGoal = toGoal;
        
        // ターゲット到達時の報酬
        centerCptCount = center.GetComponent<Center>().cptCount;
        AddReward((centerCptCount - prevCptCount) * k6);
        fitness6 += (centerCptCount - prevCptCount) * k6;
        prevCptCount = centerCptCount;
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
        float rotInput = Input.GetAxis("Horizontal");
        float redInput = 1.0f;
        float blueInput = 0.0f;
        
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
        continuousAct[2] = rotInput;
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