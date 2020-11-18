
import Wagu.Block;
import Wagu.Board;
import Wagu.Table;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import scala.Tuple2;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class WordCount {

    private static void printWordsSummary(List<Tuple2<Tuple2<String, Long>, Long>> wordList, String cat){
        System.out.println("\n"+cat);
        List<String> headers = Arrays.asList("Rank", "Word", "Frequency");
        List<List<String>> rows = new ArrayList<>();

        for(Tuple2<Tuple2<String, Long>, Long> t: wordList){
            rows.add(Arrays.asList(t._2+"",t._1._1,t._1._2+""));
        }
        Board board = new Board(45);
        Table table = new Table(board, 45, headers,rows);
        List<Integer> colAlignList = Arrays.asList(
                Block.DATA_MIDDLE_LEFT,
                Block.DATA_MIDDLE_LEFT,
                Block.DATA_MIDDLE_LEFT
        );
        table.setColAlignsList(colAlignList);
        String tableString = board.setInitialBlock(table.tableToBlocks()).build().getPreview();
        System.out.println(tableString);
    }

    private static void printCharsSummary(List<Tuple2<Tuple2<Character, Long>, Long>> charList, String cat){
        System.out.println(cat+"\n");
        List<String> headers = Arrays.asList("Rank", "Word", "Frequency");
        List<List<String>> rows = new ArrayList<>();

        for(Tuple2<Tuple2<Character, Long>, Long> t: charList){
            rows.add(Arrays.asList(t._2+"",String.valueOf(t._1._1),t._1._2+""));
        }
        Board board = new Board(45);
        String tableString = board.setInitialBlock(new Table(board, 45, headers,rows).tableToBlocks()).build().getPreview();
        System.out.println(tableString+"\n\n");
    }

    private static void printOverallSummary(long [] summary, boolean word){
        System.out.println("-".repeat(100)+"\n");
        String choice = "";
        int i = 0;
        int end = summary.length;
        if(word) {
            System.out.println("Output Summary");
            System.out.println("-".repeat(100));
            choice = "word";
        }else{
            choice = "letter";
            i = 1;
            end++;
        }
        String message = "";
        for(; i<end; i++){
            switch (i){
                case 0: if(word){ message = "Total number of words: "+summary[i]; break;}
                case 1: message = "Total number of distinct " +choice+"s: "+summary[i-1]; break;
                case 2: message = "Popular threshold "+ choice+": "+summary[i-1]; break;
                case 3: message = "Common threshold l "+ choice+": "+summary[i-1]; break;
                case 4: message = "Common threshold u "+ choice+": "+summary[i-1]; break;
                case 5: message = "Rare threshold "+ choice+": "+summary[i-1]; break;
            }
            System.out.println(message);
        }
        System.out.println("-".repeat(100)+"\n");
    }
    private static void wordCount(String fileName) {

        SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("JD Word Counter");

        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        JavaRDD<String> inputFile = sparkContext.textFile(fileName);
        final String pattern = "[a-zA-Z0-9]+";
        JavaRDD<String> wordsFromFile = inputFile.flatMap(content -> {
            Matcher matcher = Pattern.compile(pattern).matcher(content);
            ArrayList<String> list = new ArrayList<>();
            while(matcher.find()){
                list.add(matcher.group());
            }
            return list.iterator();
            //return Arrays.asList(content.split("[^a-zA-Z0-9]")).listIterator();
        });

        JavaPairRDD<String, Long> countWords = wordsFromFile
                .mapToPair(t -> {
            return new Tuple2(t, 1L);//No counting NULL byte words
        }).reduceByKey((x, y) -> (long) x + (long) y);

        JavaRDD<String> chars = countWords.flatMap(wp -> {
            char[] stringArr = wp._1.toCharArray();
            String[] transformed = new String[stringArr.length];
            for(int i=0; i<stringArr.length; i++){
                transformed[i] = String.valueOf(stringArr[i])+wp._2 + "";
            }
            return Arrays.asList(transformed).listIterator();
        });

        JavaPairRDD<Character, Long> charCount = chars.mapToPair(trs -> {
            return new Tuple2((char)trs.charAt(0),Long.parseLong(trs.substring(1)));
        }).reduceByKey((x,y) -> (long)x+(long)y);

        JavaPairRDD<Tuple2<String, Long>, Long> sortedWords = countWords
                .mapToPair(pair -> pair.swap())
                .sortByKey(false)
                .mapToPair(pair -> pair.swap())
                .zipWithIndex();

        JavaPairRDD<Tuple2<Character, Long>, Long> sortedChars = charCount
                .mapToPair(pair -> pair.swap())
                .sortByKey(false)
                .mapToPair(pair -> pair.swap())
                .zipWithIndex();

        long totalWords = wordsFromFile.count();
        long totalDistinctWords = sortedWords.count();
        long totalChars = sortedChars.count();

        long pw = (totalDistinctWords*5)/100;
        long rw = (totalDistinctWords*95)/100;
        long mu = (totalDistinctWords*525)/1000;
        long ml = (totalDistinctWords*475)/1000;

        List<Tuple2<Tuple2<String, Long>, Long>> pWords = sortedWords.filter(w -> w._2 <= pw).collect();
        List<Tuple2<Tuple2<String, Long>, Long>> mWords = sortedWords.filter(w -> w._2 <= mu && w._2 >= ml).collect();
        List<Tuple2<Tuple2<String, Long>, Long>> rWords = sortedWords.filter(w -> w._2 >= rw).collect();

        long popularWordThresh = pWords.get(pWords.size()-1)._1._2;
        long commonWordsThreshU = mWords.get(0)._1._2;
        long commonWordsThreshL = mWords.get(mWords.size()-1)._1._2;
        long rareWordThresh = rWords.get(0)._1._2;

        long pc = (totalChars*5)/100;
        long rc = (totalChars*95)/100;
        long muc = (totalChars*525)/1000;
        long mlc = (totalChars*475)/1000;

        List<Tuple2<Tuple2<Character, Long>, Long>> pChars = sortedChars.filter(c -> c._2 <= pc).collect();
        List<Tuple2<Tuple2<Character, Long>, Long>> mChars = sortedChars.filter(c -> c._2 <= muc && c._2 >= mlc).collect();
        List<Tuple2<Tuple2<Character, Long>, Long>> rChars = sortedChars.filter(c -> c._2 >= rc).collect();

        long popularCharThresh = pChars.get(pChars.size()-1)._1._2;
        long commonCharThreshU = mChars.get(0)._1._2;
        long commonCharsThreshL = mChars.get(mChars.size()-1)._1._2;
        long rareCharsThresh = rChars.get(0)._1._2;

        long wordsSummary[] = {totalWords, totalDistinctWords, popularWordThresh, commonWordsThreshL, commonWordsThreshU, rareWordThresh};
        long charsSummary[] = {totalChars, popularCharThresh, commonCharsThreshL, commonCharThreshU, rareCharsThresh};

        printOverallSummary(wordsSummary, true);
        printWordsSummary(pWords, "Popular Words");
        printWordsSummary(mWords, "Common Words");
        printWordsSummary(rWords, "Rare Words");

        printOverallSummary(charsSummary, false);
        printCharsSummary(pChars, "Popular Letters");
        printCharsSummary(mChars, "Common Letters");
        printCharsSummary(rChars, "Rare Letters");

        System.out.println("-".repeat(100)+"\n");

    }

    public static void main(String[] args) {

        if (args.length == 0) {
            System.out.println("No files provided.");
            System.exit(0);
        }

        wordCount(args[0]);
    }
}