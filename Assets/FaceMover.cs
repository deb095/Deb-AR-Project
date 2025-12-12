using UnityEngine;

public class FaceMover : MonoBehaviour
{
    public float speed = 2f;

    void Update()
    {
        // Move Left/Right
        if (Input.GetKey(KeyCode.A)) transform.Translate(-speed * Time.deltaTime, 0, 0);
        if (Input.GetKey(KeyCode.D)) transform.Translate(speed * Time.deltaTime, 0, 0);

        // Move Up/Down
        if (Input.GetKey(KeyCode.W)) transform.Translate(0, speed * Time.deltaTime, 0);
        if (Input.GetKey(KeyCode.S)) transform.Translate(0, -speed * Time.deltaTime, 0);

        // Move Forward/Backward (3D movement)
        if (Input.GetKey(KeyCode.E)) transform.Translate(0, 0, speed * Time.deltaTime);
        if (Input.GetKey(KeyCode.Q)) transform.Translate(0, 0, -speed * Time.deltaTime);

        // Rotate left/right
        if (Input.GetKey(KeyCode.LeftArrow)) transform.Rotate(0, -60 * Time.deltaTime, 0);
        if (Input.GetKey(KeyCode.RightArrow)) transform.Rotate(0, 60 * Time.deltaTime, 0);
    }
}
