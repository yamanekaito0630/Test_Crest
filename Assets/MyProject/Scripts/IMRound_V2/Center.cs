using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Center : MonoBehaviour
{
    public int cptCount;

    public void Initialize()
    {
        cptCount = 0;
    }

    public void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("CheckPoint1"))
        {
            if (cptCount % 2 == 0)
            {
                cptCount++;
            }
        }
        
        if (other.CompareTag("CheckPoint2"))
        {
            if (cptCount % 2 == 1)
            {
                cptCount++;
            }
        }
    }
}
