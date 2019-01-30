package peer.afang.dl.neuralnetwork.bp;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * @author ZhangZhenfang
 * @date 2019/1/28 20:11
 */
public class NewBp {

    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }



    public static void main(String[] args) {
        double[][] inputs = new double[][]{{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}};
        double[] labels = new double[]{0,1, 1, 0};
        Mat input = new Mat(1, 3, CvType.CV_32F);
        input.put(0, 0, inputs[0]);
        NewBp newBp = new NewBp(input, 2, new int[]{4, 3}, 1);
        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < 4; j++) {
                input = new Mat(1, 3, CvType.CV_32F);
                input.put(0, 0, inputs[j]);
                newBp.forward(input);
//                System.out.println(newBp.outLayer.getOutput().dump());
//                System.out.println(newBp.hidLayers.get(0).getWeight().dump());
//                System.out.println(newBp.hidLayers.get(0).getOutput().dump());
//                System.out.println(newBp.outLayer.getWeight().dump());
//                new Scanner(System.in).nextLine();
                Mat label = new Mat(1, 1, CvType.CV_32F);
                label.put(0, 0, labels[j]);
                newBp.back(input, label, 0.1);

            }
        }
        Mat in = new Mat(1, 3, CvType.CV_32F);
        in.put(0, 0, inputs[0]);
        newBp.forward(in);
        System.out.println(newBp.outLayer.getOutput().dump());
        in = new Mat(1, 3, CvType.CV_32F);
        in.put(0, 0, inputs[1]);
        newBp.forward(in);
        System.out.println(newBp.outLayer.getOutput().dump());
        in = new Mat(1, 3, CvType.CV_32F);
        in.put(0, 0, inputs[2]);
        newBp.forward(in);
        System.out.println(newBp.outLayer.getOutput().dump());
        in = new Mat(1, 3, CvType.CV_32F);
        in.put(0, 0, inputs[3]);
        newBp.forward(in);
        System.out.println(newBp.outLayer.getOutput().dump());

    }

    private List<HidLayer> hidLayers;
    private OutLayer outLayer;

    public NewBp(Mat input, int numberOfHidLayer, int[] lengths, int outputLength) {
        this.hidLayers = new ArrayList<HidLayer>();
        HidLayer hidLayer = new HidLayer(input, lengths[0]);
        hidLayer.print();
        hidLayers.add(hidLayer);
        for (int i = 1; i < numberOfHidLayer; i++) {
            hidLayer = new HidLayer(hidLayer.getOutput(), lengths[i]);
            hidLayer.print();
            hidLayers.add(hidLayer);
        }
        outLayer = new OutLayer(hidLayer.getOutput(), outputLength);
        outLayer.print();
    }

    public List<HidLayer> getHidLayers() {
        return hidLayers;
    }

    public void setHidLayers(List<HidLayer> hidLayers) {
        this.hidLayers = hidLayers;
    }

    public OutLayer getOutLayer() {
        return outLayer;
    }

    public void setOutLayer(OutLayer outLayer) {
        this.outLayer = outLayer;
    }

    public void forward(Mat input) {
        hidLayers.get(0).setInput(input);
        for (int i = 0; i < hidLayers.size(); i++) {
            HidLayer layer = hidLayers.get(i);
            layer.computeOut();
            sigmoid(layer.getOutput());
        }
        outLayer.computeOut();
//        System.out.println("____________________________________________________");
//        System.out.println(outLayer.getOutput().dump());
        sigmoid(outLayer.getOutput());
//        System.out.println(outLayer.getOutput().dump());
    }

    public void back(Mat input, Mat label, double rate) {
        outLayer.computeDelta(label, new Mat());
        outLayer.updateWeight(rate);
        hidLayers.get(hidLayers.size() - 1).computeDelta(outLayer.getDelta(), outLayer.getWeight());
        hidLayers.get(hidLayers.size() - 1).updateWeight(rate);
        for (int i = hidLayers.size() - 2; i >= 0; i--) {
            hidLayers.get(i).computeDelta(hidLayers.get(i + 1).getDelta(), hidLayers.get(i + 1).getWeight());
            hidLayers.get(i).updateWeight(rate);
        }
    }

    /**
     * 对矩阵的每一个元素进行sigmoid运算
     * @param m
     */
    public void sigmoid(Mat m) {
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                // 取值经过sigmoid后再放回原处
                m.put(i, j, new double[]{sigmoidFunction(m.get(i, j)[0])});
            }
        }
    }

    /**
     * sigmoid激活函数
     * @param x 输入值
     * @return
     */
    public double sigmoidFunction(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}




