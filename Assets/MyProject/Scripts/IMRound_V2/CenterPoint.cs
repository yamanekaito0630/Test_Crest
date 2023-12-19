using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CenterPoint : MonoBehaviour
{
    private GameObject[] robots;
    private Vector3 totalPos;
    private Vector3 averagePos;

    void Start()
    {
        robots = GameObject.FindGameObjectsWithTag("Robot");
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        foreach (GameObject robot in robots)
        {
            totalPos += robot.transform.position;
        }
        averagePos = totalPos / robots.Length;
        transform.position = averagePos;
        totalPos = Vector3.zero;
    }
}
