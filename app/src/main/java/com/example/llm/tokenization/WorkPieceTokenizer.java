package com.example.llm.tokenization;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class WorkPieceTokenizer {
    private final Map<String, Integer> dic;
    private static final String UNKNOWN_TOKEN = "[UNK]";
    private static final String CLS = "[CLS]";
    private static final String SEP = "[SEP]";
    private static final int EXTRA_ID_NUM = 2;
    private static final int MAX_INPUTCHARS_PER_WORD = 100;

    public WorkPieceTokenizer(Map<String,Integer> vocab){ dic = vocab; }

    /**
     * Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first
     * algorithm to perform tokenization using the given vocabulary. For example: input = "unaffable"
     * output = ["un", "##aff" , "##able"]
     *
     * @param text: A single token or whitespace separated tokens. This should have already been
     *            passed through 'BasicTokenizer'.
     * @return A list of wordpiece tokens.
     */
    public List<String> tokenize(String text) {
        if (text==null){
            throw new NullPointerException("The input string is null!!!");
        }
        List<String> outputTokens = new ArrayList<>();
        for(String token: BasicTokenizer.whitespaceTokenize(text)){
            if(token.length() > MAX_INPUTCHARS_PER_WORD){
                outputTokens.add(UNKNOWN_TOKEN);
                continue;
            }

            boolean isBad = false; // Mark if a word cannot be tokenized into known subwords
            int start = 0;
            List<String> subTokens = new ArrayList<>();

            while(start< token.length()){
                String curSubStr = "";

                int end = token.length(); // Longer substring matches first.
                while(start<end){
                    String subStr =
                            (start==0) ? token.substring(start,end) : "##" + token.substring(start,end);
                    if(dic.containsKey(subStr)){
                        curSubStr = subStr;
                        break;
                    }
                    end--;
                }

                // the word doesn't contain any known subwords
                if(curSubStr.isEmpty()){
                    isBad = true;
                    break;
                }

                // curSubStr is the longest subword that can be found
                subTokens.add(curSubStr);

                // Proceed to tokenize the resident string
                start = end;
            }

            if(isBad){
                outputTokens.add(UNKNOWN_TOKEN);
            }
            else{
                outputTokens.addAll(subTokens);
            }
        }
        return outputTokens;
    }

    public List<String> addSpecialToken(List<String> tokens, int maxLen){
        List<String> outputTokens = new ArrayList<>();

        int inputLen = Math.min(maxLen-EXTRA_ID_NUM, tokens.size());
        outputTokens.add(CLS);
        outputTokens.addAll(tokens.subList(0, inputLen));
        outputTokens.add(SEP);

        return outputTokens;
    }
}
