package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import peer.afang.dl.neuralnetwork.bp.Bp;
import peer.afang.dl.neuralnetwork.bp.NewBp;
import peer.afang.dl.neuralnetwork.bp.OutLayer;
import peer.afang.dl.util.MnistReader;
import peer.afang.dl.util.MnistTest;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * @author ZhangZhenfang
 * @date 19/1/23 14:43
 */
public class Cnn {

    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }

    private ConvLayer convLayer1;
    private ConvLayer convLayer2;

    private PoolingLayer poolingLayer1;
    private PoolingLayer poolingLayer2;

    private Mat bpInput;
    private NewBp newBp;

    public static void main(String[] args) throws Exception {
        MnistReader reader = new MnistReader(MnistReader.TRAIN_IMAGES_FILE, MnistReader.TRAIN_LABELS_FILE);
        double[] nextImage = reader.getNextImage(true);
        Mat m = new Mat(28, 28, CvType.CV_32F);
        m.put(0, 0, nextImage);
//        m.reshape(28);
        List<Mat> input = new ArrayList();
        input.add(m);
        Cnn cnn = new Cnn(input);
        cnn.forward(input);
        Mat nextLabel = MnistTest.getNextLabel(reader);
        cnn.back(nextLabel);
        reader.open(MnistReader.TRAIN_IMAGES_FILE, MnistReader.TRAIN_LABELS_FILE);
        while(true) {
            input.clear();
            nextImage = reader.getNextImage(true);
            if (nextImage == null) {
                break;
            }
            Mat in = new Mat(28, 28, CvType.CV_32F);
            in.put(0, 0, nextImage);
            input.add(in);
            cnn.forward(input);
            nextLabel = MnistTest.getNextLabel(reader);
            cnn.back(nextLabel);
            System.out.println(cnn.convLayer1.getDelta().get(0).dump());
            new Scanner(System.in).nextLine();
//            new Scanner(System.in).nextLine();
        }
        test(cnn);

    }

    public static void test(Cnn bp) {
        MnistReader testMnistReader = new MnistReader(MnistReader.TEST_IMAGES_FILE, MnistReader.TEST_LABELS_FILE);
        int numberOfTest = 0;
        int right = 0;
        int error = 0;
        while(true) {
            double[] nextImage = testMnistReader.getNextImage(true);
            if (nextImage == null) {
                break;
            }
            Mat input = new Mat(28, 28, CvType.CV_32F);
            input.put(0, 0, nextImage);
            List<Mat> list = new ArrayList<Mat>();
            list.add(input);
            bp.forward(list);

            Mat label = MnistTest.getNextLabel(testMnistReader);
            OutLayer outLayer = bp.newBp.getOutLayer();
            int[] predict = MnistTest.indexOfMax(outLayer.getOutput());
            int[] real = MnistTest.indexOfMax(label);
            numberOfTest++;
            if (predict[0] == real[0] && predict[1] == real[1]) {
                right++;
            } else {
                error++;
            }
        }
        System.out.print(numberOfTest + " " + right + " " + error + "  ");
        System.out.println("correct ratio : " + (double) right / numberOfTest);
    }

    public Cnn(List<Mat> input) {
        convLayer1 = new ConvLayer(input, 5, 0, 1, 1);
        poolingLayer1 = new PoolingLayer(convLayer1.getOut(), 2, 0, 2, 1);
        convLayer2 = new ConvLayer(poolingLayer1.getOut(), 3, 0, 1, 1);
        poolingLayer2 = new PoolingLayer(convLayer2.getOut(), 2, 0, 2, 1);
        bpInput = new Mat(poolingLayer2.getOut().size(), 25, CvType.CV_32F);
        newBp = new NewBp(bpInput, 2, new int[]{20, 15}, 10);

    }

    public void forward(List<Mat> input) {
        convLayer1.setInput(input);
        convLayer1.computeOut();
        poolingLayer1.computeOut();
        convLayer2.computeOut();
        poolingLayer2.computeOut();

        List<Mat> out1 = poolingLayer2.getOut();

        List<Mat> mats = new ArrayList<Mat>();

        for (int i = 0; i < out1.size(); i++) {
            mats.add(out1.get(i).reshape(1, 1));
        }
        Core.vconcat(mats, bpInput);
        bpInput = bpInput.reshape(1, 1);
        newBp.forward(bpInput);
    }

    public void back(Mat label) {
        newBp.back(bpInput, label, 0.1);
        Mat delta = new Mat(1, 25, CvType.CV_32F);
        Core.gemm(newBp.getHidLayers().get(0).getDelta(), newBp.getHidLayers().get(0).getWeight().t(), 1, new Mat(), 1, delta);
        delta = delta.reshape(1, 1);

        for (int i = 0; i < delta.rows(); i++) {
            poolingLayer2.getDelta().clear();
            poolingLayer2.getDelta().add(delta.row(i).reshape(1, poolingLayer2.getOut().get(0).rows()));
        }
        convLayer2.computeDeltaFromPoolingLayer(poolingLayer2.getDelta(), poolingLayer2.getPosition());
        convLayer2.updateFilter();
        poolingLayer1.computeDeltaFromConvLayer(convLayer2.getDelta(), convLayer2.getFilters().get(0));
        convLayer1.computeDeltaFromPoolingLayer(poolingLayer1.getDelta(), poolingLayer1.getPosition());
        convLayer1.updateFilter();
    }

    public static void maxPooling(Mat src, int rows, int cols, Mat dst, Mat pos) throws Exception {
        double[] data = new double[dst.cols() * dst.rows()];
        double[] positionData = new double[dst.cols() * dst.rows()];
        int i = 0;
        if (src.rows() % rows != 0 || src.cols() % cols != 0) {
            System.out.println(src);
            System.out.println(rows + " " + cols);
            throw new Exception("size don't fit");
        }
        if (src.rows() / rows != dst.rows() || src.cols() / cols != dst.cols()) {
            throw new Exception("dst size don't fit");
        }
        for (int r = 0; r < src.rows(); r += rows) {
            for (int c = 0; c < src.cols();  c += cols) {
                Mat mat = src.rowRange(r, r + rows).colRange(c, c + cols);
                Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(mat);
                positionData[i] = minMaxLocResult.maxLoc.x + minMaxLocResult.maxLoc.y * mat.cols();
                data[i++] = minMaxLocResult.maxVal;
            }
        }
        pos.put(0, 0, positionData);
        dst.put(0, 0, data);
    }

    public static void relu(Mat mat) {
        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                // 取值经过relu后再放回原处
                mat.put(i, j, new double[]{reluFunction(mat.get(i, j)[0])});
            }
        }
    }
    public static double reluFunction(double x) {
        return x > 0 ? x : 0;
    }
}