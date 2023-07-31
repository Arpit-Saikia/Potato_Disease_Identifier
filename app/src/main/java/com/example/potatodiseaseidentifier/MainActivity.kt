package com.example.potatodiseaseidentifier

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.potatodiseaseidentifier.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    public lateinit var gallery_Button: Button
    public lateinit var camera_Button: Button
    public lateinit var result: TextView
    public lateinit var image: ImageView

    val imageSize = 256

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        gallery_Button = findViewById(R.id.button2)
        camera_Button = findViewById(R.id.button)

        result = findViewById(R.id.result)
        image = findViewById(R.id.imageView)

        camera_Button.setOnClickListener {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) run {
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, 3)
            } else {
                requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
            }
        }
        gallery_Button.setOnClickListener {
            val galleryIntent =
                Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(galleryIntent, 1)
        }


    }

    public fun classifyImage(image: Bitmap) {
        val model = Model.newInstance(applicationContext)

// Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.FLOAT32)

        var byteBuffer =
            ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3) //4 bytes with l=b=32 and 3 channels as RGB
        byteBuffer.order(ByteOrder.nativeOrder());
        val intValues = IntArray(imageSize * imageSize)
        image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
        var pixel = 0
        //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
        //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
        for (i in 0 until imageSize) {
            for (j in 0 until imageSize) {
                val value = intValues[pixel++] // RGB
                byteBuffer.putFloat((value shr 16 and 0xFF) * (1f / 1))
                byteBuffer.putFloat((value shr 8 and 0xFF) * (1f / 1))
                byteBuffer.putFloat((value and 0xFF) * (1f / 1))
            }
        }
        inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        val confidence = outputFeature0.floatArray
        var maxConfidence = 1f*0
        var maxPos = -1
        for (i in confidence.indices) {
            if (confidence.get(i) > maxConfidence) {
                maxConfidence  = confidence.get(i)
                maxPos = i
            }
        }
        val classes = arrayOf("Potato Early Blight", "Potato Late Blight", "Potato Healthy")
        result.text = classes[maxPos]
// Releases model resources if no longer used.
        model.close()

    }

    //launching the camera or the gallery
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) {
                var imageSet = data!!.extras!!["data"] as Bitmap
                classifyImage(imageSet)
                val dimension = Math.min(imageSet!!.width, imageSet.height)
                imageSet = ThumbnailUtils.extractThumbnail(imageSet, dimension, dimension)
                image.setImageBitmap(imageSet)
                classifyImage(imageSet)
            } else {
                var dat = data!!.data;
                var imageSet: Bitmap? = null
                imageSet = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                image.setImageBitmap(imageSet)
                imageSet = Bitmap.createScaledBitmap(imageSet, imageSize, imageSize, false)
                classifyImage(imageSet)
            }

        }

        super.onActivityResult(requestCode, resultCode, data)

    }
}
