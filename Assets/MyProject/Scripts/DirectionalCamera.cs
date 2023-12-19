using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DirectionalCamera : MonoBehaviour
{
    public bool isDetactRed;
    public bool isDetactBlue;
    
    public void OnTriggerStay(Collider other)
    {
        if (other.CompareTag("RedLED"))
        {
            isDetactRed = true;
        }
        if (other.CompareTag("BlueLED"))
        {
            isDetactBlue = true;
        }
    }
    
    public void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("RedLED"))
        {
            isDetactRed = false;
        }
        if (other.CompareTag("BlueLED"))
        {
            isDetactBlue = false;
        }
    }
}

