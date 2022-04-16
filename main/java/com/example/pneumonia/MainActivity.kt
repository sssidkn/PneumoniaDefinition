package com.example.pneumonia

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.example.pneumonia.ml.Model
import org.tensorflow.lite.support.image.TensorImage
import java.lang.Math.round


class MainActivity : AppCompatActivity() {
    private lateinit var imageBitMap: Bitmap
    private lateinit var resultView: TextView

    companion object {
        private const val IMAGE_CHOOSE = 1000
        private const val PERMISSION_CODE = 1001
        private const val REQUEST_IMAGE = 1002
    }

    private fun chooseImageGallery() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = "image/*"
        startActivityForResult(intent, IMAGE_CHOOSE)
    }

    @RequiresApi(Build.VERSION_CODES.M)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val open_cam = findViewById<ImageButton>(R.id.btn_open_cam)
        val getImage = findViewById<ImageButton>(R.id.getImage)
        val getResult = findViewById<ImageButton>(R.id.getResult)
        val open_history = findViewById<Button>(R.id.hist_btn)
        val save_btn = findViewById<Button>(R.id.save_btn)
        resultView = findViewById(R.id.resultView)
        open_cam.setOnClickListener {
            val CameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
                val permissions = arrayOf(Manifest.permission.CAMERA)
                requestPermissions(permissions, PERMISSION_CODE)
            } else {
                startActivityForResult(CameraIntent, REQUEST_IMAGE)
            }
        }
        getImage.setOnClickListener {
            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED) {
                val permissions = arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
                requestPermissions(permissions, PERMISSION_CODE)
            } else {
                chooseImageGallery()
            }
        }
        getResult.setOnClickListener {
            outputGenerator(imageBitMap)
        }
        open_history.setOnClickListener {
            val intent = Intent(this@MainActivity, HistoryActivity::class.java)
            startActivity(intent)
        }
        save_btn.setOnClickListener {

        }
    }

    @SuppressLint("SetTextI18n")
    private fun outputGenerator(bitmap: Bitmap) {
        val pneumoniaDefinition = Model.newInstance(this)
        val newBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)

        val tfImage = TensorImage.fromBitmap(newBitmap)
        val output = pneumoniaDefinition.process(tfImage)
            .probabilityAsCategoryList.apply {
                sortByDescending { it.score }
            }
        val res = output[0]
        val con = round(output[0].score * 100)
        resultView.text = "${res.label} $con%"
        pneumoniaDefinition.close()

    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val imageView = findViewById<ImageView>(R.id.ImageView)
        if (requestCode == REQUEST_IMAGE && resultCode == RESULT_OK) {
            imageBitMap = data?.extras?.get("data") as Bitmap
            imageView.setImageBitmap(imageBitMap)
            resultView.text = "RESULT"
        } else if (requestCode == IMAGE_CHOOSE && resultCode == RESULT_OK) {
            val imageUri = data?.data
            imageBitMap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri)
            imageView.setImageBitmap(imageBitMap)
            resultView.text = "RESULT"
        }
        super.onActivityResult(requestCode, resultCode, data)
    }
}

