using System.Net.Mail;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.PlayerLoop;

public class Player_V2 : Agent
{
    // 各種パラメータ
    public float propulsion = 30.0f;
    public float defaultDrag = 0.0f;
    public float waterDrag = 10.0f;
    public Vector3 initPos;

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

    private Rigidbody playerRb;
    private Rigidbody rightFrontFanRb;
    private Rigidbody leftFrontFanRb;
    private Rigidbody rightMiddleFanRb;
    private Rigidbody leftMiddleFanRb;
    private Rigidbody rightBehindFanRb;
    private Rigidbody leftBehindFanRb;
    
    private int cptCount;

    private GameObject goal1;
    private float toGoal1;
    private float prevToGoal1;
    
    private GameObject goal2;
    private float toGoal2;
    private float prevToGoal2;

    private bool isAboveOcean;

    public override void Initialize()
    {
        playerRb = GetComponent<Rigidbody>();
        
        rightFrontFanRb = rightFrontFan.GetComponent<Rigidbody>();
        leftFrontFanRb = leftFrontFan.GetComponent<Rigidbody>();
        rightMiddleFanRb = rightMiddleFan.GetComponent<Rigidbody>();
        leftMiddleFanRb = leftMiddleFan.GetComponent<Rigidbody>();
        rightBehindFanRb = rightBehindFan.GetComponent<Rigidbody>();
        leftBehindFanRb = leftBehindFan.GetComponent<Rigidbody>();

        goal1 = GameObject.FindWithTag("CheckPoint1");
        goal2 = GameObject.FindWithTag("CheckPoint2");
        prevToGoal1 = Vector3.Distance(goal1.transform.position, transform.position);
        prevToGoal2 = Vector3.Distance(goal2.transform.position, transform.position);

        isAboveOcean = true;
    }

    public override void OnEpisodeBegin()
    {
        // 初期化処理
        transform.position = initPos;
        transform.rotation = Quaternion.identity;
        cptCount = 0;

        // エピソード開始時に水中ドローンに初期推進力を与える
        // playerRb.AddForce(transform.TransformDirection(new Vector3(0, 200.0f, 200.0f)));
    }

    public void OnTriggerEnter(Collider other)
    {
        // ゴール到達時の報酬の計算
        if (other.CompareTag("CheckPoint1"))
        {
            if (cptCount % 2 == 0)
            {
                AddReward(1.0f);
                cptCount++;
            }
        }
        if (other.CompareTag("CheckPoint2"))
        {
            if (cptCount % 2 == 1)
            {
                AddReward(1.0f);
                cptCount++;
            }
        }
    }
    
    public void OnTriggerStay(Collider other)
    {
        // タグが"Obstacle"のオブジェクトに衝突している場合
        if (other.CompareTag("Obstacle"))
        {
            // 衝突ペナルティの計算
            AddReward(-0.01f);
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

    public override void OnActionReceived(ActionBuffers actions)
    {
        // アクションを取得
        float rightFrontInput = actions.ContinuousActions[0];
        float leftFrontInput = actions.ContinuousActions[1];
        float rightMiddleInput = actions.ContinuousActions[2];
        float leftMiddleInput = actions.ContinuousActions[3];
        float rightBehindInput = actions.ContinuousActions[4];
        float leftBehindInput = actions.ContinuousActions[5];

        if (isAboveOcean)
        {
            rightFrontInput = 0.0f;
            leftFrontInput = 0.0f;
            rightMiddleInput = 0.0f;
            leftMiddleInput = 0.0f;
            rightBehindInput = 0.0f;
            leftBehindInput = 0.0f;
        }
        

        // 各ファンにForceを適用
        rightFrontFanRb.AddRelativeForce(new Vector3(0.0f, rightFrontInput * propulsion, 0.0f));
        leftFrontFanRb.AddRelativeForce(new Vector3(0.0f, leftFrontInput * propulsion, 0.0f));
        rightMiddleFanRb.AddRelativeForce(new Vector3(0.0f, rightMiddleInput * propulsion, 0.0f));
        leftMiddleFanRb.AddRelativeForce(new Vector3(0.0f, leftMiddleInput * propulsion, 0.0f));
        rightBehindFanRb.AddRelativeForce(new Vector3(0.0f, rightBehindInput * propulsion, 0.0f));
        leftBehindFanRb.AddRelativeForce(new Vector3(0.0f, leftBehindInput * propulsion, 0.0f));

        // ゴールとの距離
        toGoal1 = Vector3.Distance(goal1.transform.position, transform.position);
        toGoal2 = Vector3.Distance(goal2.transform.position, transform.position);
        
        // 距離報酬の計算
        if (cptCount % 2 == 0)
        {
            AddReward(prevToGoal1 - toGoal1);
        }
        else if(cptCount % 2 == 1)
        {
            AddReward(prevToGoal2 - toGoal2);
        }
        
        // ゴールとの距離を上書き
        prevToGoal1 = toGoal1;
        prevToGoal2 = toGoal2;
    }

    // 手動テスト用
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // デフォルトの入力
        float leftFrontInput = Input.GetKey(KeyCode.Q) ? 1.0f : 0.0f;
        float leftMiddleInput = Input.GetKey(KeyCode.A) ? 1.0f : 0.0f;
        float leftBehindInput = Input.GetKey(KeyCode.Z) ? 1.0f : 0.0f;
        float rightFrontInput = Input.GetKey(KeyCode.W) ? 1.0f : 0.0f;
        float rightMiddleInput = Input.GetKey(KeyCode.S) ? 1.0f : 0.0f;
        float rightBehindInput = Input.GetKey(KeyCode.X) ? 1.0f : 0.0f;

        // 入力をエージェントのアクションに割り当て
        ActionSegment<float> continuousAct = actionsOut.ContinuousActions;
        continuousAct[0] = rightFrontInput;
        continuousAct[1] = leftFrontInput;
        continuousAct[2] = rightMiddleInput;
        continuousAct[3] = leftMiddleInput;
        continuousAct[4] = rightBehindInput;
        continuousAct[5] = leftBehindInput;
    }
}