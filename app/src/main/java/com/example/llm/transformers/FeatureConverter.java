package com.example.llm.transformers;

import com.example.llm.tokenization.FullTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


/*
* Convert String to features that can be fed into BERT model.
* **/
public class FeatureConverter {
    private final FullTokenizer tokenizer;
    private final int maxSeqLen;
    private final boolean padToMaxLength;

    public FeatureConverter(Map<String, Integer> inputDic, boolean doLowerCase, int maxSeqLen, boolean padToMaxLength){
        this.tokenizer = new FullTokenizer(inputDic, doLowerCase);
        this.maxSeqLen = maxSeqLen;
        this.padToMaxLength = padToMaxLength;
    }

    public Feature convert(String text, boolean addSpecialTokens){
        List<String> tokens = tokenizer.tokenize(text);
        if(addSpecialTokens){
            tokens = tokenizer.addSpecialToken(tokens, maxSeqLen);
        }

        List<Integer> inputIds = tokenizer.convertTokensToIds(tokens);
        if(padToMaxLength){
            while (inputIds.size()> maxSeqLen){
                inputIds.add(0);
            }
        }
        return new Feature(inputIds);
    }

    public Feature convert(String text){
        text = convertWords(text);
        String[] words = text.split(" ");
        List<String> tokens = new ArrayList<>();
        for (String word: words){
            if(!word.equals("")){
                tokens.add(word);
            }
        }
        if(tokens.size()>maxSeqLen-1){
            tokens = tokens.subList(0, maxSeqLen-1);
        }

        List<Integer> inputIds = tokenizer.convertTokensToIds(tokens);
        if(padToMaxLength){
            while (inputIds.size()> maxSeqLen){
                inputIds.add(0);
            }
        }
        return new Feature(inputIds);
    }

    String convertWords(String sentence){
        List<String> newWords = new ArrayList<>();
        Pattern pattern = Pattern.compile("(?i)\\b\\w\\w+\\b");
        Matcher matcher = pattern.matcher(sentence.toLowerCase());
        while (matcher.find()){
            String word = matcher.group();
            newWords.add(word);
        }
        return String.join(" ", newWords);
    }
}
