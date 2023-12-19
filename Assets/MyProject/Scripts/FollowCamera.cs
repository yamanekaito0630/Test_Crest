using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FollowCamera : MonoBehaviour
{
    public GameObject followRobot;
    public bool attachX;
    public bool attachY;
    public bool attachZ;

    public float deltaX = 0.0f;
    public float deltaY = 0.0f;
    public float deltaZ = 0.0f;
    
    private Vector3 robotPosition;
    private Vector3 myPosition; 

    // Update is called once per frame
    void Update()
    {
        robotPosition = followRobot.transform.position;
        myPosition = transform.position;
        if (attachX)
        {
            myPosition.x = robotPosition.x + deltaX;
        }

        if (attachY)
        {
            myPosition.y = robotPosition.y + deltaY;
        }

        if (attachZ)
        {
            myPosition.z = robotPosition.z + deltaZ;
        }
        transform.position = myPosition;
    }
}
