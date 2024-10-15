package com.example.llm.tokenization;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/*
* A java realization of Bert tokenization. Original python code:
* https://github.com/google-research/bert/blob/master/tokenization.py run full tokenization to
* tokenize a String into split subtokes or ids.
* **/
public class FullTokenizer {
    private final BasicTokenizer basicTokenizer;
    private final WorkPieceTokenizer workPieceTokenizer;
    private final Map<String, Integer> dic;

    public FullTokenizer(Map<String, Integer> inputDic, boolean doLowerCase){
        dic = inputDic;
        basicTokenizer = new BasicTokenizer(doLowerCase);
        workPieceTokenizer = new WorkPieceTokenizer(inputDic);
    }

    public List<String> tokenize(String text){
        List<String> splitTokens = new ArrayList<>();
        for(String token : basicTokenizer.tokenize(text)){
            splitTokens.addAll(workPieceTokenizer.tokenize(token));
        }
        return splitTokens;
    }

    public List<Integer> convertTokensToIds(List<String> tokens){
        List<Integer> outputIds = new ArrayList<>();
        for(String token: tokens){
            if(dic.get(token) != null){
                outputIds.add(dic.get(token));
            }
        }
        return outputIds;
    }

    public List<String> addSpecialToken(List<String> tokens, int maxLen){
        return workPieceTokenizer.addSpecialToken(tokens, maxLen);
    }
}
