using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LEDs : MonoBehaviour
{
    void Update()
    {
        if (this.GetComponent<Renderer>().material.color == Color.red)
        {
            this.tag = "RedLED";
        }
        else if (this.GetComponent<Renderer>().material.color == Color.blue)
        {
            this.tag = "BlueLED";
        }
        else
        {
            this.tag = "Untagged";
        }
    }
}
