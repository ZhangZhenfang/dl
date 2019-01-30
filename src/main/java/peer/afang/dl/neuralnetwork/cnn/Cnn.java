package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import peer.afang.dl.neuralnetwork.bp.NewBp;
import peer.afang.dl.util.MnistReader;
import peer.afang.dl.util.MnistTest;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 19/1/23 14:43
 */
public class Cnn {

    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }
    private Mat weight;

    public static void main(String[] args) throws Exception {
        MnistReader reader = new MnistReader(MnistReader.TRAIN_IMAGES_FILE, MnistReader.TRAIN_LABELS_FILE);
        double[] nextImage = reader.getNextImage(true);
        Mat m = new Mat(28, 28, CvType.CV_32F);
        m.put(0, 0, nextImage);
        m.reshape(28);
        List<Mat> input = new ArrayList();
        input.add(m);

        ConvLayer convLayer1 = new ConvLayer(input, 3, 1, 1, 1);
        convLayer1.computeOut();
        convLayer1.print();

        PoolingLayer poolingLayer1 = new PoolingLayer(convLayer1.getOut(), 2, 0, 2, 1);
        poolingLayer1.computeOut();
        poolingLayer1.print();
        List<Mat> out = poolingLayer1.getOut();

        ConvLayer convLayer2 = new ConvLayer(out, 3, 1, 1, 1);
        convLayer2.computeOut();
        convLayer2.print();

        PoolingLayer poolingLayer2 = new PoolingLayer(convLayer2.getOut(), 2, 0, 2, 1);
        poolingLayer2.computeOut();
        poolingLayer2.print();

        List<Mat> out1 = poolingLayer2.getOut();
        Mat bpInput = new Mat(out1.size(), 49, CvType.CV_32F);
        List<Mat> mats = new ArrayList<Mat>();

        for (int i = 0; i < out1.size(); i++) {
            mats.add(out1.get(i).reshape(1, 1));
        }
        Core.vconcat(mats, bpInput);
        System.out.println("111" + bpInput);
        bpInput = bpInput.reshape(1, 1);
        NewBp newBp = new NewBp(bpInput, 2, new int[]{20, 15}, 10);
//        Bp bp = new Bp(bpInput.cols(), 2, new int[]{20, 15}, 10);
//        bp.newForward(bpInput);
        newBp.forward(bpInput);
        System.out.println(newBp.getOutLayer().getOutput().dump());
        double[] nextLabel = reader.getNextLabel();
        Mat label = MnistTest.getNextLabel(reader);
        newBp.back(bpInput, label, 0.1);
        System.out.println(newBp.getOutLayer().getDelta().dump());
        System.out.println(newBp.getHidLayers().get(1).getDelta().dump());
        System.out.println(newBp.getHidLayers().get(0).getDelta().dump());
        Mat delta = new Mat(1, 147, CvType.CV_32F);
        System.out.println(newBp.getHidLayers().get(0).getWeight());
        Core.gemm(newBp.getHidLayers().get(0).getDelta(), newBp.getHidLayers().get(0).getWeight().t(), 1, new Mat(), 1, delta);
//        Mat delta = newBp.getHidLayers().get(0).getDelta().reshape(1, poolingLayer2.getDelta().size());
        delta = delta.reshape(1, 1);
        for (int i = 0; i < delta.rows(); i++) {
            System.out.println(delta);
            System.out.println(poolingLayer2.getOut().get(0));
            poolingLayer2.getDelta().add(delta.row(i).reshape(1, poolingLayer2.getOut().get(0).rows()));
        }
        convLayer2.computeDeltaFromFC(poolingLayer2.getDelta(), poolingLayer2.getPosition());
        System.out.println(convLayer2.getDelta().get(0).dump());
        poolingLayer1.computeDeltaFromConvLayer(convLayer2.getDelta(), convLayer2.getFilters().get(0));
        convLayer1.computeDeltaFromFC(poolingLayer1.getDelta(), poolingLayer1.getPosition());
        System.out.println(convLayer1.getDelta().get(0).dump());
//        Mat delta = newBp.getHidLayers().get(0).getDelta();
//        System.out.println(delta.dump());
    }

    public Cnn() {
        this.weight = new Mat(3, 3, CvType.CV_32F);
        double[] data = new double[weight.cols() * weight.rows()];
        for (int i = 0; i < data.length; i++) {
            data[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        weight.put(0, 0, data);
    }

    public void forward(Mat mat) throws Exception{
        Mat conDst = new Mat(mat.size(), CvType.CV_32F);
        Imgproc.filter2D(mat, conDst, mat.depth(), weight);
        Mat poolDst = new Mat(conDst.rows() / 2, conDst.cols() / 2, CvType.CV_32F);
        Mat positionDst = new Mat(conDst.rows() / 2, conDst.cols() / 2, CvType.CV_16U);
        maxPooling(conDst, 2, 2, poolDst, positionDst);
        System.out.println(conDst.dump());
        System.out.println(poolDst.dump());
        System.out.println(positionDst.dump());
    }

    public void print() {
        System.out.println("weight:");
        System.out.println(weight.dump());
    }

    public static void maxPooling(Mat src, int rows, int cols, Mat dst, Mat pos) throws Exception {
        double[] data = new double[dst.cols() * dst.rows()];
        double[] positionData = new double[dst.cols() * dst.rows()];
        int i = 0;
        if (src.rows() % rows != 0 || src.cols() % cols != 0) {
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