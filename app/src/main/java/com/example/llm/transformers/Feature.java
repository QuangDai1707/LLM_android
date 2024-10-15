package com.example.llm.transformers;

import com.google.common.primitives.Ints;

import java.util.List;

public class Feature {
    public int[] inputIds;

    public Feature(List<Integer> inputIds){
        this.inputIds = Ints.toArray(inputIds);
    }
}

