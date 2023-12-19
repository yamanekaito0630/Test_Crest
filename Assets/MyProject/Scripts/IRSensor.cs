using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class IRSensor : MonoBehaviour
{
    public bool isDetact = false;

    void OnTriggerStay(Collider other)
    {
        if (other.CompareTag("Robot"))
        {
            isDetact = true;
        }
    }
    
    void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Robot"))
        {
            isDetact = false;
        }
    }
}
