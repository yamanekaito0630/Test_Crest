using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Windows;

public class ScreenShotCaputurer : MonoBehaviour
{
    public string directory;
    [SerializeField] private Camera _camera;
    public int depth = 24;
    private int count = 0;

    private void Update()
    {
        name = Application.persistentDataPath + "/" + directory + "/img_" + count.ToString() + ".png";
        // スクリーンショットを保存
        CaptureScreenShot(name);
        
        // メモリ解放
        if (count % 500 == 0)
        {
            System.GC.Collect();
            Resources.UnloadUnusedAssets();
        }
        count++;
    }

    private void CaptureScreenShot(string filePath)
    {
        var rt = new RenderTexture(_camera.pixelWidth, _camera.pixelHeight, depth);
        var prev = _camera.targetTexture;
        _camera.targetTexture = rt;
        _camera.Render();
        _camera.targetTexture = prev;
        RenderTexture.active = rt;

        var screenShot = new Texture2D(
            _camera.pixelWidth,
            _camera.pixelHeight,
            TextureFormat.RGB24,
            false
        );
        screenShot.ReadPixels(new Rect(0, 0, screenShot.width, screenShot.height), 0, 0);
        screenShot.Apply();

        var bytes = screenShot.EncodeToPNG();
        Destroy(screenShot);
        System.IO.File.WriteAllBytes(filePath, bytes);
    }
}