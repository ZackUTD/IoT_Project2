import com.google.gson.Gson;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import org.influxdb.BatchOptions;
import org.influxdb.InfluxDB;
import org.influxdb.InfluxDBFactory;
import org.influxdb.dto.BatchPoints;
import org.influxdb.dto.Point;
import org.influxdb.dto.Pong;

import util.Config;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

public class Generator {
    private String db = "bearing";
    private static int BW = 100; // buffer window
    private static int MW = BW * 5; // merge window
    private static int mw_count = 0; // merge window counter
    private static int FILE_SIZE = 20;
    private static int INTERVAL = 1;

    private static int LABEL_BASELINE = 0;
    private static int LABEL_OUTER_RACE_FAULT = 1;
    private static int LABEL_INNER_RACE_FAULT = 2;

    private static String ATTR_SR = "sr";
    private static String ATTR_GS = "gs";
    private static String ATTR_LOAD = "load";
    private static String ATTR_RATE = "rate";

    private String[] filenames = new String[] {
        "mats" + File.separator + "baseline_1.mat",
        "mats" + File.separator + "baseline_2.mat",
        "mats" + File.separator + "baseline_3.mat",

        "mats" + File.separator + "OuterRaceFault_1.mat",
        "mats" + File.separator + "OuterRaceFault_2.mat",
        "mats" + File.separator + "OuterRaceFault_3.mat",

        "mats" + File.separator + "InnerRaceFault_vload_1.mat",
        "mats" + File.separator + "InnerRaceFault_vload_2.mat",
        "mats" + File.separator + "InnerRaceFault_vload_3.mat",
        "mats" + File.separator + "InnerRaceFault_vload_4.mat",
        "mats" + File.separator + "InnerRaceFault_vload_5.mat",
        "mats" + File.separator + "InnerRaceFault_vload_6.mat",
        "mats" + File.separator + "InnerRaceFault_vload_7.mat",

        "mats" + File.separator + "OuterRaceFault_vload_1.mat",
        "mats" + File.separator + "OuterRaceFault_vload_2.mat",
        "mats" + File.separator + "OuterRaceFault_vload_3.mat",
        "mats" + File.separator + "OuterRaceFault_vload_4.mat",
        "mats" + File.separator + "OuterRaceFault_vload_5.mat",
        "mats" + File.separator + "OuterRaceFault_vload_6.mat",
        "mats" + File.separator + "OuterRaceFault_vload_7.mat"
    };

    private MLStructure[] readers = new MLStructure[FILE_SIZE];
    private int[] lengths = new int[FILE_SIZE];
    private double[] probs = new double[FILE_SIZE];
    private int totalLength;
    private Random random = new Random();
    private int[] cursors = new int[FILE_SIZE];
    private long timestamp;
    private Gson gson;
    private long interArrivalTime;

    // Added InfluxDB Variables
    private String host = "http://localhost:8086";
    private String username = "test"; // We may not need username and password
    private String password = "test1";
    private String output_db_offline = "team_3_test_offline";
    private String output_db_online = "team_3_test_online";
    private InfluxDB influxDB;
    private BatchPoints batchPoints;

    public Generator() {
        gson = new Gson();
        interArrivalTime = Config.getInstance().getInterArrivalTime();
    }

    private void loadMatFiles() throws IOException {
        totalLength = 0;
        for (int i = 0; i < filenames.length; i++) {
            MatFileReader reader = new MatFileReader(filenames[i]);
            readers[i] = (MLStructure) reader.getMLArray(db);
            int len = readers[i].getField(ATTR_GS).getSize();
            lengths[i] = len - len % MW;
            totalLength += lengths[i];
            cursors[i] = 0;
        }

        probs[0] = lengths[0] * .1 / totalLength;
        for (int i = 1; i < probs.length; i++) {
            probs[i] = probs[i - 1] + lengths[i] * .1 / totalLength;
        }

        System.out.println("Total Length: " + Integer.toString(totalLength));
    }

    private int selectFile() {
        double prob = random.nextDouble();
        int min, max;

        if (prob < 0.7) {
            min = 0;
            max = 2;
        }
        else {
            min = 3;
            max = 19;
        }

        return ThreadLocalRandom.current().nextInt(min, max + 1);
    }

    private void connect(String output_db) {
        try{
            influxDB = InfluxDBFactory.connect(host);
            influxDB.setDatabase(output_db);
            influxDB.enableBatch(BatchOptions.DEFAULTS.actions(BW).exceptionHandler(
                    (failedPoints, throwable) -> throwable.printStackTrace()
            ));

            batchPoints = BatchPoints.database(output_db).build();

            Pong response = this.influxDB.ping();
            if (response.getVersion().equalsIgnoreCase("unknown")) {
                System.out.println("Error pinging server in connect.");
            }
            else {
                System.out.println("InfluxDB ping successful");
            }
        }
        catch (Exception e) {
            System.out.println("ERROR: Failed to connect");
            System.out.println(e.getMessage());
            System.exit(1);
        }
    }

    private int generatePoints(int index, String metric) {
        int len = readers[index].getField(ATTR_GS).getSize();
        int start = cursors[index];
        int end = cursors[index] + MW;
        if (MW > len - cursors[index]) {
            cursors[index] = 0;
        }

        for (int i = start; i < end; i++ ) {
            // Old point creation code for model.Point class. Now using influxDB Point class
//            Point point = new Point().withMetric(metric)
//                                .withTimestamp(timestamp++)
//                                .withLabel(getLabel(index))
//                                .withSr(((MLDouble)readers[index].getField(ATTR_SR)).getArray()[0][0])
//                                .withRate(((MLDouble)readers[index].getField(ATTR_RATE)).getArray()[0][0])
//                                .withGs(((MLDouble)readers[index].getField(ATTR_GS)).getArray()[i][0]);
            String load;
            if (readers[index].getField(ATTR_LOAD) instanceof MLChar) {
//                point.setLoad(((MLChar) readers[index].getField(ATTR_LOAD)).getString(0));
                load = ((MLChar) readers[index].getField(ATTR_LOAD)).getString(0);
            }
            else {
//                point.setLoad(String.valueOf(((MLDouble) readers[index].getField(ATTR_LOAD)).getArray()[0][0]));
                load = String.valueOf(((MLDouble) readers[index].getField(ATTR_LOAD)).getArray()[0][0]);
            }

            Point point = Point.measurement("bearing").time(timestamp++, TimeUnit.NANOSECONDS)
                    .addField("label", getLabel(index))
                    .addField(ATTR_SR, ((MLDouble)readers[index].getField(ATTR_SR)).getArray()[0][0])
                    .addField("rate", ((MLDouble)readers[index].getField(ATTR_RATE)).getArray()[0][0])
                    .addField("gs", ((MLDouble)readers[index].getField(ATTR_GS)).getArray()[i][0])
                    .addField("load", load)
                    .addField("mw", mw_count)
                    .build();

            output(point);
        }

        return MW;
    }

    private void output(Point point) {

        if(influxDB != null) {
            //add to buffer
            batchPoints.point(point);
            //if buffer is full, write the entire buffer in a
            if(batchPoints.getPoints().size() >= BW) {
                try {
                    influxDB.write(batchPoints);
                    System.out.println("Sending batch");
                    batchPoints = BatchPoints.database(batchPoints.getDatabase()).build();
                    System.out.println(batchPoints.getPoints().size());
                }
                catch (Exception e) {
                    System.out.println("ERROR: Failed sending batch");
                    System.out.println(e.getMessage());
                    System.exit(1);
                }
            }

        }
        else {
            System.out.println(gson.toJson(point));
        }
    }

    private int getLabel(int index) {
        if (index < 3)
            return LABEL_BASELINE;
        else if (index < 6)
            return LABEL_OUTER_RACE_FAULT;
        else if (index < 13)
            return LABEL_INNER_RACE_FAULT;
        else
            return LABEL_OUTER_RACE_FAULT;
    }

    private int getInterArrivalOffset() {
        return (random.nextInt(21) - 10) * 10;
    }

    public void startOffline() {
        int offlineCount = 0;
        int offlineTotal = totalLength / 2;
        timestamp = 0;
        while (offlineCount < offlineTotal) {
            int index = selectFile();
            offlineCount += generatePoints(index, "offline");
            mw_count++;
            try {
                Thread.sleep(interArrivalTime + getInterArrivalOffset());
            } catch (InterruptedException ignored) {}
        }
        // TODO: send whatever's left in the buffer to the DB in one last batch?
    }

    public void startOnline() {
        timestamp = 0;
        while (true) {
            int index = selectFile();
            generatePoints(index, "online");
            mw_count++;
            try {
                Thread.sleep(interArrivalTime + getInterArrivalOffset());
            } catch (InterruptedException ignored) {}
        }
    }

    private static String usage = "Usage: datagen [-offline | -online | -all] [-influxdb] [-host <insert host>]";

    public static void main(String[] args) {
        Generator generator = new Generator();

        try {
            generator.loadMatFiles();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        if (args.length > 0) {
            boolean offline = false;
            boolean online = false;
            boolean influxdb = false;
            boolean hostGiven = false; // whether the host was given
            boolean hostFlag = false; // whether the -host flag was just read. Used to get the host on the next iteration
            for (String arg: args) {
                System.out.println(arg);

                if(hostFlag) {
                   generator.host = arg;
                   hostFlag = false;
                   hostGiven = true;
                   continue;
                }

                switch (arg) {
                    case "-offline":
                        if(online || offline) {
                            System.out.println(usage);
                            return;
                        }
                        offline = true;
                        break;
                    case "-online":
                        if(online || offline) {
                            System.out.println("Usage: datagen [-offline | -online | -all] [-influxdb]");
                            return;
                        }
                        online = true;
                        break;
                    case "-all":
                        if(online || offline) {
                            System.out.println("Usage: datagen [-offline | -online | -all] [-influxdb]");
                            return;
                        }
                        offline = true;
                        online = true;
                        break;
                    case "-influxdb":
                        if(influxdb) {
                            System.out.println("Usage: datagen [-offline | -online | -all] [-influxdb]");
                            return;
                        }
                        influxdb = true;
                        break;
                    case "-host":
                        if(hostGiven || hostFlag) {
                            System.out.println("Usage: datagen [-offline | -online | -all] [-influxdb]");
                            return;
                        }
                        hostFlag = true;
                        break;
                    default:
                        System.out.println("Usage: datagen [-offline | -online | -all] [-influxdb]");
                        return;
                }
            }

            if(offline) {
                if(influxdb) {
                    generator.connect(generator.output_db_offline);
                }
                generator.startOffline();
            }

            if(online) {
                if(influxdb) {
                    generator.connect(generator.output_db_online);
                }
                generator.startOnline();
            }
        }
        else {
            generator.startOffline();
            generator.startOnline();
        }
    }
}
