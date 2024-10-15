package com.example.llm;

import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

import androidx.activity.EdgeToEdge;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class BaseModuleActivity extends AppCompatActivity {
    private static final int UNSET = 0;

    protected HandlerThread mBackgroundThread;
    protected Handler mBackgroundHandler;
    protected Handler mUIHandler;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mUIHandler = new Handler(getMainLooper());
    }

    @Override
    protected void onPostCreate(@Nullable Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);
        final Toolbar toolbar = findViewById(R.id.toolbar);
        if (toolbar != null){
            setSupportActionBar(toolbar);
        }
        startBackgroundThread();
    }

    @Override
    protected void onDestroy() {
        stopBackgroundThread();
        super.onDestroy();
    }

    protected void startBackgroundThread(){
        mBackgroundThread = new HandlerThread("ModuleActivity");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    protected void stopBackgroundThread(){
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            Log.e("SpamClassification", "Error in stopping background thread", e);
        }
    }
}