package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import peer.afang.dl.util.Sigmoid;

/**
 * @author ZhangZhenfang
 * @date 2019/2/27 22:04
 */
public class BP {

    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }
    private InputLayer inputLayer;
    private HidLayer hidLayer1;
    private HidLayer hidLayer2;
    private OutLayer outLayer;

    public static void main(String[] args) {
        double[][] inputs = new double[][]{{0, 0}, {0, 1}, {1, 1}, {1, 0}};
        double[] labels = new double[]{0, 1, 1, 0};
        Mat input = new Mat(1, 2, CvType.CV_32F);
        input.put(0, 0, inputs[0]);
        BP bp = new BP();

        System.out.println(bp.hidLayer1.weight.dump());
        System.out.println(bp.hidLayer2.weight.dump());
        System.out.println(bp.outLayer.weight.dump());
        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < 4; j++) {
                input = new Mat(1, 2, CvType.CV_32F);
                input.put(0, 0, inputs[j]);
                bp.forword(input);
                Mat label = new Mat(1, 1, CvType.CV_32F);
                label.put(0, 0, labels[j]);
                bp.back(label, 0.1);
            }
        }
        Mat in = new Mat(1, 2, CvType.CV_32F);
        in.put(0, 0, inputs[0]);
        bp.forword(in);
        System.out.println(bp.outLayer.getA().dump());
        in = new Mat(1, 2, CvType.CV_32F);
        in.put(0, 0, inputs[1]);
        bp.forword(in);
        System.out.println(bp.outLayer.getA().dump());
        in = new Mat(1, 2, CvType.CV_32F);
        in.put(0, 0, inputs[2]);
        bp.forword(in);
        System.out.println(bp.outLayer.getA().dump());
        in = new Mat(1, 2, CvType.CV_32F);
        in.put(0, 0, inputs[3]);
        bp.forword(in);
        System.out.println(bp.outLayer.getA().dump());
    }

    private BP() {
        this.inputLayer = new InputLayer();
        this.hidLayer1 = new HidLayer(2, 4, new Sigmoid());
        this.hidLayer2 = new HidLayer(4, 3, new Sigmoid());
        this.outLayer = new OutLayer(3, 1, new Sigmoid());
//        Mat m = new Mat(3, 4, CvType.CV_32F);
//        m.put(0, 0, TestData.weight1);
//        hidLayer1.setWeight(m);
//        m = new Mat(4, 3, CvType.CV_32F);
//        m.put(0, 0, TestData.weight2);
//        hidLayer2.setWeight(m);
//        m = new Mat(3, 1, CvType.CV_32F);
//        m.put(0, 0, TestData.weight3);
//        outLayer.setWeight(m);

        hidLayer1.setPre(inputLayer);
        hidLayer2.setPre(hidLayer1);
        outLayer.setPre(hidLayer2);
        hidLayer1.setNext(hidLayer2);
        hidLayer2.setNext(outLayer);
    }

    public void forword(Mat input) {
        inputLayer.setA(input);
        hidLayer1.computeOut();
        hidLayer2.computeOut();
        outLayer.computeOut();
    }

    public void back(Mat label, double rate) {
        outLayer.computeGrad(label);
        hidLayer2.computeGrad();
        hidLayer1.computeGrad();
        outLayer.updateWeight(rate);
        hidLayer2.updateWeight(rate);
        hidLayer1.updateWeight(rate);
    }
}