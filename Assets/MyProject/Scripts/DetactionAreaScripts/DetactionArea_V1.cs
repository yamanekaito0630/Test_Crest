using System.Collections.Generic;
using UnityEngine;

public class DetactionArea_V1 : MonoBehaviour
{
    public GameObject mySelf;
    private List<GameObject> nearbyObjects = new List<GameObject>();
    public Vector2 firstNeighborhood;
    public Vector2 secondNeighborhood;
    
    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Robot"))
        {
            nearbyObjects.Add(other.gameObject);
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Robot"))
        {
            nearbyObjects.Remove(other.gameObject);
        }
    }

    void Update()
    {
        if (nearbyObjects.Count == 1)
        {
            firstNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V1>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V1>().nodeIndex
            );
            secondNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V1>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V1>().nodeIndex
            );
        }
        else if (nearbyObjects.Count == 2)
        {
            firstNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V1>().nodeIndex,
                (nearbyObjects[1]).GetComponent<PlayerIM_Round_V1>().nodeIndex
            );
            secondNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V1>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V1>().nodeIndex
            );
        }
        else
        {
            // 距離でソート
            nearbyObjects.Sort((a, b) => Vector3.Distance(transform.position, a.transform.position)
                .CompareTo(Vector3.Distance(transform.position, b.transform.position)));
            firstNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V1>().nodeIndex,
                (nearbyObjects[1]).GetComponent<PlayerIM_Round_V1>().nodeIndex
                );
            secondNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V1>().nodeIndex,
                (nearbyObjects[2]).GetComponent<PlayerIM_Round_V1>().nodeIndex
                );
        }
    }
}
