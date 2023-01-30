using UnityEngine;
using System.Collections;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine.Networking;

public class UdpSocket : MonoBehaviour
    /*
     Ȧ�η��� IP Address Ȯ�� �׽�Ʈ��.
     */
{
    [HideInInspector] public bool isTxStarted = false;

    string IP;
    [SerializeField] int rxPort = 3000; // port to receive data from Python on
    [SerializeField] int txPort = 3001; // port to send data to Python on
    // Google Spread Sheet���� ip �ּ� �����͸� �޾ƿͼ� ����. 
    private string url = "https://script.google.com/macros/s/AKfycbwN8eSbtq9myXTFcw8bL2i2N6R21Yr-2m41S9rdZFCMpi1tltsl1zblvpeTWNmHlKNy/exec";

    private string message = "Test_hololens";

    // Create necessary UdpClient objects
    UdpClient client;
    IPEndPoint remoteEndPoint;
    Thread receiveThread; // Receiving Thread

    IEnumerator SendDataCoroutine()
    {
        while (true)
        {
            SendData(message);
            yield return new WaitForSeconds(0.5f);
        }
    }

    public void SendData(string message) // Use to send data to Python
    {
        try
        {
            Debug.Log(message);
            byte[] data = Encoding.UTF8.GetBytes(message);
            client.Send(data, data.Length, remoteEndPoint);
        }
        catch (Exception err)
        {
            print(err.ToString());
        }
    }

    private void Awake()
    {
        Debug.Log("Awake");
        UnityWebRequest www = UnityWebRequest.Get(url);
        www.SendWebRequest();
        while (!www.isDone)
        {
            // Google Spread Sheet���� �����͸� �޾ƿ´�. �� �޾ƿ��� ��� ���� revision �ʿ�. 
            IP = www.downloadHandler.text;
        }
        Debug.Log(IP);
    }

    void Start()
    {
        // Create remote endpoint (to Matlab) 
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(IP), txPort);

        // Initialize (seen in comments window)
        print("UDP Comms Initialised");
    }

    public void GetIP()
    {
        //�޼����� ���̽� ������ ����
        Debug.Log("press");
        Debug.Log(IP);
        receiveThread.Start();
        StartCoroutine(SendDataCoroutine());
    }
}
