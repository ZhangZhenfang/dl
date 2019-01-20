package peer.afang.dl.neural_net_work;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;

import java.util.List;
import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 2019/1/20 0:21
 */
public class BP {
    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }
    private Mat inputLayer;
    private Mat hiddenLayer;
    private Mat outputLayer;
    private Mat lables;

    private Mat weights1;
    private Mat weights2;

    public BP(int inputLength, int inputNum, int hidden, int out) {
        this.inputLayer = new Mat(inputNum, inputLength, CvType.CV_32F);
        this.inputLayer.put(0, 0, new double[]{0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1});
        this.hiddenLayer = new Mat(hidden, 1, CvType.CV_32F);
        this.outputLayer = new Mat(out, 1, CvType.CV_32F);
        this.weights1 = new Mat(inputLength, hidden, CvType.CV_32F);
        this.weights2 = new Mat(hidden, out, CvType.CV_32F);
        this.lables = new Mat(inputNum, 1, CvType.CV_32F);
        double[] v = new double[inputLength * hidden];
        for (int i = 0; i < v.length; i++) {
            v[i] = new Random().nextDouble();
        }
        this.weights1.put(0, 0, v);
        v = new double[hidden * out];
        for (int i = 0; i < v.length; i++) {
            v[i] = new Random().nextDouble();
        }
        this.weights2.put(0, 0, v);
        this.lables.put(0, 0, new double[]{0, 1, 1, 0});

        System.out.println("input:\n" + inputLayer.dump());
        System.out.println("hidden:\n" + hiddenLayer.dump());
        System.out.println("output:\n" + outputLayer.dump());
        System.out.println("weight1:\n" + weights1.dump());
        System.out.println("weights2:\n" + weights2.dump());
        System.out.println("lables:\n" + lables.dump());
    }

    public void forward() {
        for (int i = 0; i < this.inputLayer.size().height; i++) {
            Core.gemm(this.inputLayer.row(i), this.weights1, 1, new Mat(), 1, this.hiddenLayer);
            Core.gemm(this.hiddenLayer, this.weights2, 1, new Mat(), 1, this.outputLayer);
            System.out.println(this.hiddenLayer.dump());
            System.out.println(this.outputLayer.dump());
            backPropagation(this.lables.row(i));
        }

    }
    public void backPropagation(Mat lable) {
        Mat one = Mat.ones(outputLayer.size(), CvType.CV_32F);
        Mat t1 = new Mat(outputLayer.size(), CvType.CV_32F);
        Mat t2= new Mat(outputLayer.size(), CvType.CV_32F);
        Core.subtract(one, outputLayer, t1);
        Core.multiply(outputLayer, t1, t1);
        Core.subtract(lable, outputLayer, t2);
        Core.multiply(t1, t2, t1);
        System.out.println("t1:\n" + t1.dump());
        Mat t3 = new Mat(weights2.size(), CvType.CV_32F);
        System.out.println(hiddenLayer + "\n" + t1 + "\n" + t3);
        Core.gemm(hiddenLayer.t(), t1.t(), 1, new Mat(), 1, t3);
        Core.add(t3, weights2, weights2);
        System.out.println("t3:\n" + t3.dump());
    }
    public static void main(String[] args) {
        BP bp = new BP(3, 4, 4, 1);
        bp.forward();
    }

    public void mul(double[][] left, double[][] right) {
        if (left == null || right == null) {
            return ;
        }
    }
}
