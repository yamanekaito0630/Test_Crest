using System.Collections.Generic;
using UnityEngine;

public class DetactionArea : MonoBehaviour
{
    public GameObject mySelf;
    private List<GameObject> nearbyObjects = new List<GameObject>();
    public Vector2 firstNeighborhood;
    public Vector2 secondNeighborhood;
    public Vector2 thirdNeighborhood;
    public Vector2 fourthNeighborhood;
    public Vector2 fifthNeighborhood;
    public Vector2 sixthNeighborhood;
    
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
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            secondNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            thirdNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fourthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fifthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            sixthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
        }
        else if (nearbyObjects.Count == 2)
        {
            firstNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[1]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            secondNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            thirdNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fourthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fifthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            sixthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
        }
        else if (nearbyObjects.Count == 3)
        {
            firstNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[1]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            secondNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[2]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            thirdNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fourthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fifthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            sixthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
        }
        else if (nearbyObjects.Count == 4)
        {
            firstNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[1]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            secondNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[2]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            thirdNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[3]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fourthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fifthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            sixthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
        }
        else if (nearbyObjects.Count == 5)
        {
            firstNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[1]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            secondNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[2]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            thirdNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[3]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fourthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[4]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fifthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            sixthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
        }
        else if (nearbyObjects.Count == 6)
        {
            firstNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[1]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            secondNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[2]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            thirdNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[3]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fourthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[4]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fifthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[5]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            sixthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[0]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
        }
        else
        {
            // 距離でソート
            nearbyObjects.Sort((a, b) => Vector3.Distance(transform.position, a.transform.position)
                .CompareTo(Vector3.Distance(transform.position, b.transform.position)));
            firstNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[1]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            secondNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[2]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            thirdNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[3]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fourthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[4]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            fifthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[5]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
            sixthNeighborhood = new Vector2(
                mySelf.GetComponent<PlayerIM_Round_V4>().nodeIndex,
                (nearbyObjects[6]).GetComponent<PlayerIM_Round_V4>().nodeIndex
            );
        }
    }
}
