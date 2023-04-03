/*
ENTER YOUR NAME HERE
NAME: NISHITA DUTTA
MATRICULATION NUMBER: A0242122Y
*/
import java.io.IOException;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class TopkCommonWords {

    public static Integer k = 10;
    public static HashSet<String> hashSet = new HashSet<>();
    public static HashSet<String> filenames = new HashSet<>();

    public static class StopWordsMapper
            extends Mapper<Object, Text, Text, IntWritable>{
        private Text word = new Text();

        /**
         * This mapper only stores all of the stopwords into a hashmap for every mapper to access and lookup.
         * @param key
         * @param value
         * @param context
         */
        public void map(Object key, Text value, Context context
        ) {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                hashSet.add(word.toString());
            }
        }
    }

    public static class CommonWordsMapper
            extends Mapper<Object, Text, Text, Text>{
        private Text word = new Text();
        private Text file = new Text();

        /**
         * This mapper takes the two input files and emits (word, file1) or (word, file2) to indicate which word
         * came from which file. Note that this will work with any number of files.
         * @param key
         * @param value
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit)context.getInputSplit();
            //get file name
            String filename = fileSplit.getPath().getName();
            filenames.add(filename);
            file.set(filename);

            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                if(!hashSet.contains(word.toString()) && word.toString().length() > 4)
                    //emit (word, filename) where the filename is the file name where the word came from.
                    context.write(word, file);
            }
        }
    }

    public static class CommonWordsReducer
            extends Reducer<Text,Text,IntWritable, Text> {
        private IntWritable result = new IntWritable();

        /**
         * This reducer output the count of common words in the given files.
         * Note that this will work with any number of files.
         * @param key : The word
         * @param values : Iterable of filenames it came from (f1, f1, f2, f1, f1, f2)..
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        public void reduce(Text key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            HashMap<String, Integer> frequency = new HashMap<>();

            //Populate the map with the frequency of the filenames it appears in.
            for (Text val : values) {
                frequency.putIfAbsent(val.toString(), 0);
                frequency.put(val.toString(), frequency.get(val.toString()) + 1);
            }
            //frequency now looks like: {(f1: 5), (f2: 3)}
            // Which means the word appears in f1, 5 times and in f2, 3 times

	    //Get the minimum number of times it appears in any file, return without emitting if 0
	    OptionalInt min = OptionalInt.of(0);
            if(frequency.keySet().size() > 1)
                min = frequency.values().stream().mapToInt(v -> v).min();
            if(min.getAsInt() == 0) return;
	    
            //emit count, word (3, word)
            result.set(min.getAsInt());
            context.write(result, key);
        }
    }

    public static class TopkMapper extends Mapper<Object, Text, IntWritable, Text>{

        public static TreeMap<Integer, List<String>> treeMap = new TreeMap<>(Collections.reverseOrder());

        /**
         * This mapper outputs the top k words in a given chunk.
         * @param key
         * @param value every line looks like (count word)
         * @param context
         */
        public void map(Object key, Text value, Context context
        ) {
            String line=value.toString();
            String[] tokens=line.split("\t");
            Integer count = Integer.valueOf(tokens[0]);
            String word = tokens[1];
            //insert into a treemap which stores the count and list of words with that count.
            //Treemap takes care of inserting in descending sorted order, so we dont need to maintain the sorting
            //manually ourselves.
            insertIntoTreeMap(treeMap, count, word);

            //if Map contains more than 10 counts, we remove the word with least number of counts.
            //Note this takes care of words with the same count as well.
            //This is because the treemap is {'count' vs 'list of words with the same count'}
            resizeValueMapToK(treeMap);
        }

        protected void cleanup(Context context) throws IOException, InterruptedException {
            //finally emit all the content in the treemap
            for (Map.Entry<Integer, List<String>> entry :treeMap.entrySet()) {
                List<String> words = entry.getValue();
                //Sort the list of words because we want the output
                // to have the words of the same count in sorted order
                Collections.sort(words);
                Collections.reverse(words);
                for(String word : words) {
                    context.write(new IntWritable(entry.getKey()), new Text(word));
                }
            }
        }
    }


    public static class TopkReducer
            extends Reducer<IntWritable, Text, IntWritable, Text> {
        public static TreeMap<Integer, List<String>> map = new TreeMap<>();

        /**
         * Reducer calculates the top k from a reduced list of items.
         * @param key count
         * @param values list of words with that count
         * @param context
         */
        public void reduce(IntWritable key, Iterable<Text> values,
                           Context context
        ) {
            for(Text word: values) {
                insertIntoTreeMap(map, key.get(), word.toString());
            }
            //only save the top k counts
            resizeValueMapToK(map);
        }

        protected void cleanup(Context context) throws IOException, InterruptedException {
            int i = 0;
            //iterate the map in descending order.
            for (Integer count : map.descendingKeySet()) {
                List<String> words = map.get(count);
                Collections.sort(words);
                for(String word : words) {
                    i++;
                    //if we have output k words already then the rest of the words
                    // in the map are not required. Break
                    if(i > k) break;
                    context.write(new IntWritable(count), new Text(word));
                }
            }
        }
    }

    //Helper methods
    public static void insertIntoTreeMap(TreeMap<Integer, List<String>> tMap, Integer count, String word) {
        if(tMap.containsKey(count)) {
            tMap.get(count).add(word);
        }else{
            ArrayList<String> list = new ArrayList<>();
            list.add(word);
            tMap.put(count, list);
        }
    }

    private static void resizeValueMapToK(TreeMap<Integer, List<String>> tmap) {
        while(tmap.size()> k){
            tmap.pollLastEntry();
        }
    }

    public static void main(String[] args) throws Exception{
        k = Integer.valueOf(args[4]);
        Path inputFile1Path = new Path(args[0]);
        Path inputFile2Path = new Path(args[1]);
        Path stopWordsFilePath = new Path(args[2]);

        String interDirString = "/home/course/cs4225/cs4225_assign/temp/assign1_inter/A0242122Y";
        //String interDirString = "data";
        Path outputPathOfStopper = new Path(interDirString + "/output_intermediate");
        Path outputPathOfCommon  = new Path(interDirString + "/output_intermediate_1");

        Path inputPathOfTopk = outputPathOfCommon;
        Path outPathFinal = new Path(args[3]);

        Job jobCommon = Job.getInstance(new Configuration(), "common");
        Job jobSaveStopWords = Job.getInstance(new Configuration(), "stop words");
        Job jobTopk = Job.getInstance(new Configuration(), "topk");

        jobSaveStopWords.setJarByClass(TopkCommonWords.class);
        jobSaveStopWords.setMapperClass(StopWordsMapper.class);
        jobSaveStopWords.setOutputKeyClass(Text.class);
        jobSaveStopWords.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(jobSaveStopWords, stopWordsFilePath);
        FileOutputFormat.setOutputPath(jobSaveStopWords, outputPathOfStopper);

        jobSaveStopWords.waitForCompletion(true);

        jobCommon.setJarByClass(TopkCommonWords.class);
        jobCommon.setMapperClass(CommonWordsMapper.class);
        jobCommon.setReducerClass(CommonWordsReducer.class);
        jobCommon.setOutputKeyClass(Text.class);
        jobCommon.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(jobCommon, inputFile1Path);
        FileInputFormat.addInputPath(jobCommon, inputFile2Path);
        FileOutputFormat.setOutputPath(jobCommon, outputPathOfCommon);

        jobCommon.waitForCompletion(true);


        jobTopk.setJarByClass(TopkCommonWords.class);
        jobTopk.setMapperClass(TopkMapper.class);
        jobTopk.setReducerClass(TopkReducer.class);
        jobTopk.setNumReduceTasks(1);
        jobTopk.setOutputKeyClass(IntWritable.class);
        jobTopk.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(jobTopk, inputPathOfTopk);
        FileOutputFormat.setOutputPath(jobTopk, outPathFinal);

        FileSystem fs = FileSystem.get(new Configuration());

        boolean hasCompleted = jobTopk.waitForCompletion(true);

        fs.delete(outputPathOfCommon, true);
        fs.delete(outputPathOfStopper, true);
        System.exit(hasCompleted ? 0 : 1);

    }
}
