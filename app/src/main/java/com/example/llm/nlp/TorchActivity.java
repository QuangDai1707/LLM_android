package com.example.llm.nlp;

import android.os.Bundle;
import android.os.SystemClock;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.widget.EditText;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.appcompat.widget.Toolbar;

import com.example.llm.BaseModuleActivity;
import com.example.llm.R;
import com.example.llm.transformers.Feature;
import com.example.llm.transformers.FeatureConverter;
import com.example.llm.view.ResultRowView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class TorchActivity extends BaseModuleActivity {
    private static final String TAG = "SpamClassification";
    private static final String MODEL_PATH = "DistilBert.onnx";
    private static final String DIC_PATH = "vocab.txt";
    private static final Integer SHAPE_SIZE = 512;
    private static final long EDIT_TEXT_STOP_DELAY = 600L;
    private static final String FORMAT_MS = "%dms";
    private static final String SCORE_FORMAT = "%.2f";
    private static final boolean IS_LLM_MODEL = true;

    private EditText mEditText;
    private View mResultContent;
    private ResultRowView[] mResultRowViews = new ResultRowView[3]; // Positive & Negative & Time elapsed
    private Toolbar toolbar;
    private String mLastBgHandledText;

    private Map<String, Integer> dic = new HashMap<>();
    private static final int MAX_SEQ_LEN = 512;
    private static final boolean DO_LOWER_CASE = false;
    private static final boolean PAD_TO_MAX_LENGTH = false;

    private static final boolean ADD_SPECIAL_TOKENS = true;
    private FeatureConverter featureConverter;

    private OrtEnvironment env;
    private OrtSession session;

    public void loadDictionary() {
        try (InputStream is = getAssets().open(DIC_PATH);
             BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            int index = 0;
            while (reader.ready()) {
                String key = reader.readLine();
                dic.put(key, index++);
            }
        } catch (IOException e) {
            Log.e(TAG, Objects.requireNonNull(e.getMessage()));
        }
    }

    public void loadModel(){
        long startTime = SystemClock.elapsedRealtime();
        env = OrtEnvironment.getEnvironment();
        Log.v(TAG, "getEnv: "+ (SystemClock.elapsedRealtime()-startTime));

        try (InputStream is = getAssets().open(MODEL_PATH)){
            int size = is.available();
            byte[] buffer = new byte[size];

            int read;
            while ((read = is.read(buffer, 0, size)) != -1){
                if(read < 0 ){
                    break;
                }
            }

            startTime = SystemClock.elapsedRealtime();
            session = env.createSession(buffer);
            Log.v(TAG, "CreateSession: " + (SystemClock.elapsedRealtime() - startTime));
        } catch (OrtException | IOException e) {
            throw new RuntimeException(e);
        }
    }


    private static class AnalysisResult {
        private final float[] scores;
        private final String[] className;
        private final long moduleForwardDuration;

        public AnalysisResult(float[] scores, long moduleForwardDuration){
            this.scores = scores;
            this.moduleForwardDuration = moduleForwardDuration;
            this.className = new String[2];
            this.className[0] = "Normal";
            this.className[1] = "Spam";
        }
    }

    private Runnable mOnEditTextStopRunnable = () -> {
        final String text = mEditText.getText().toString();
        mBackgroundHandler.post(()->{
            if(TextUtils.equals(text, mLastBgHandledText)){
                return;
            }

            if(TextUtils.isEmpty(text)){
                runOnUiThread(()-> applyUIEmptyTextState());
                mLastBgHandledText = null;
                return;
            }

            final AnalysisResult result = analyzeText(text);
            if(result != null){
                runOnUiThread(()-> applyUIAnalysisResult(result));
                mLastBgHandledText = text;
            }
        });
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.v(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_base_module);
        mEditText = findViewById(R.id.nsmc_edit_text);
        findViewById(R.id.nsmc_clear_button).setOnClickListener(v-> mEditText.setText(""));

        toolbar = findViewById(R.id.toolbar);
        toolbar.setTitle(R.string.pytorch);

        final ResultRowView headerRow = findViewById(R.id.nsmc_result_header_row);
        headerRow.nameTextView.setText(R.string.sentiment);
        headerRow.scoreTextView.setText(R.string.score);
        headerRow.setVisibility(View.VISIBLE);

        mResultRowViews[0] = findViewById(R.id.nsmc_top1_result_row);
        mResultRowViews[1] = findViewById(R.id.nsmc_top2_result_row);
        mResultRowViews[2] = findViewById(R.id.nsmc_time_row);
        mResultContent = findViewById(R.id.nsmc_result_content);

        mEditText.addTextChangedListener(new InternalTextWatcher());
    }

    @Override
    protected void onStart() {
        Log.v(TAG, "onStart");
        super.onStart();
        Log.v(TAG, "Loading model...");
        loadModel();
        Log.v(TAG, "Loading Dictionary...");
        loadDictionary();
        Log.v(TAG, "Loading Feature Converter");
        featureConverter = new FeatureConverter(dic, DO_LOWER_CASE, MAX_SEQ_LEN, PAD_TO_MAX_LENGTH);
    }

    private AnalysisResult analyzeText(String text){
        if(IS_LLM_MODEL){
            return analyzeTextLLM(text);
        }
        else {
            return analyzeTextMachineLearning(text);
        }
    }
    @WorkerThread
    @Nullable
    private AnalysisResult analyzeTextLLM(String text){
        Feature feature = featureConverter.convert(text, ADD_SPECIAL_TOKENS);
        int seqLength = feature.inputIds.length;
        long[] inputIds = new long[seqLength];
        for (int i = 0; i < seqLength; i++) {
            inputIds[i] = feature.inputIds[i];
        }

        Map<String, OnnxTensor> inputs = new HashMap<>();
        try {
            inputs.put((String) session.getInputNames().toArray()[0],
                    OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), new long[]{1, seqLength}));
        }
        catch (OrtException e){
            throw new RuntimeException(e);
        }
        OrtSession.Result outputs = null;
        long start = SystemClock.elapsedRealtime();
        try {
            outputs = session.run(inputs);
        }
        catch (OrtException e){
            throw new RuntimeException(e);
        }

        OnnxTensor onnxTensor = (OnnxTensor) outputs.get(0);
        float[] logits = onnxTensor.getFloatBuffer().array();
        float[] scores = new float[2];
        float sum = (float) (Math.exp(logits[0])+Math.exp(logits[1]));
        scores[0] = (float) Math.exp(logits[0])/sum;
        scores[1] = (float) Math.exp(logits[1])/sum;

        long time = SystemClock.elapsedRealtime() - start;
        return new AnalysisResult(scores, time);
    }

    private OnnxTensor createTensor(long[] input, long[] shape) throws OrtException {
        return OnnxTensor.createTensor(env, LongBuffer.wrap(input), shape);
    }

    @WorkerThread
    @Nullable
    private AnalysisResult analyzeTextMachineLearning(String text) {return null;};

    private void applyUIAnalysisResult(AnalysisResult result){
        int first_idx, second_idx;

        if(result.scores[0] <= result.scores[1]){
            // spam
            first_idx = 1;
            second_idx = 0;
        }
        else {
            first_idx = 0;
            second_idx = 1;
        }

        setUIResultRowView(
                mResultRowViews[0],
                result.className[first_idx],
                String.format(Locale.US, SCORE_FORMAT, result.scores[first_idx])
        );
        setUIResultRowView(
                mResultRowViews[1],
                result.className[second_idx],
                String.format(Locale.US, SCORE_FORMAT, result.scores[second_idx])
        );
        setUIResultRowView(
                mResultRowViews[2],
                "Time elapsed",
                String.format(Locale.US, FORMAT_MS, result.moduleForwardDuration)
        );
        mResultContent.setVisibility(View.VISIBLE);
    }

    private void applyUIEmptyTextState() {mResultContent.setVisibility(View.GONE);}

    private void setUIResultRowView(ResultRowView resultRowView, String name, String score){
        resultRowView.nameTextView.setText(name);
        resultRowView.scoreTextView.setText(score);
        resultRowView.setProgressState(false);
    }

    @Override
    protected void onStop() {
        Log.v(TAG, "onStop");
        super.onStop();
        if(session != null){
            Log.v(TAG, "Unload session...");
            try {
                session.close();
            }
            catch (Exception e){
                Log.v(TAG, "Exception: " + e);
            }
        }

        if(dic != null){
            Log.v(TAG, "Unload dictionary...");
            dic.clear();
        }
    }

    private class InternalTextWatcher implements TextWatcher{

        @Override
        public void beforeTextChanged(CharSequence s, int start, int count, int after) {

        }

        @Override
        public void onTextChanged(CharSequence s, int start, int before, int count) {

        }

        @Override
        public void afterTextChanged(Editable s) {
            mUIHandler.removeCallbacks(mOnEditTextStopRunnable);
            mUIHandler.postDelayed(mOnEditTextStopRunnable, EDIT_TEXT_STOP_DELAY);
        }
    }
}
