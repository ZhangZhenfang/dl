package peer.afang.dl.util;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * @author ZhangZhenfang
 * @date 19/1/22 9:46
 */
public class MnistReader {

    public static final String TRAIN_IMAGES_FILE = "E:\\dl\\target\\classes\\data/mnist/train-images-idx3-ubyte";
    public static final String TRAIN_LABELS_FILE = "E:\\dl\\target\\classes\\data/mnist/train-labels-idx1-ubyte";
    public static final String TEST_IMAGES_FILE = "E:\\dl\\target\\classes\\data/mnist/t10k-images-idx3-ubyte";
    public static final String TEST_LABELS_FILE = "E:\\dl\\target\\classes\\data/mnist/t10k-labels-idx1-ubyte";

    private BufferedInputStream imageBin;
    private BufferedInputStream labelBin;
    private int imageNumber;
    private int imageIndex = 0;
    private int labelNumber;
    private int labelIndex = 0;
    private int xPixel;
    private int yPixel;

    public static void main(String[] args) {
        MnistReader mnistReader = new MnistReader(TRAIN_IMAGES_FILE, TRAIN_LABELS_FILE);
        int i = 0;
        while(true) {
            double[] nextImage = mnistReader.getNextImage(false);
            if (nextImage == null) {
                break;
            }
            double[] nextLabel = mnistReader.getNextLabel();
            System.out.println(Arrays.toString(nextImage));
            System.out.println(Arrays.toString(nextLabel));
            System.out.println(i++);
        }
    }

    public MnistReader(String images, String labels) {
        open(images, labels);
    }

    /**
     * 读取下一个图像数据
     * @param norm 是否归一化
     * @return
     */
    public double[] getNextImage(boolean norm) {
        double[] element = new double[this.xPixel * this.yPixel];
        try {
            if (imageIndex >= imageNumber) {
                this.imageBin.close();
                this.labelBin.close();
                return null;
            }
            for (int j = 0; j < xPixel * yPixel; j++) {
                if (norm) {
                    element[j] = (imageBin.read() / 255.0 - 0.5) * 2;
                } else {
                    element[j] = imageBin.read();
                }
            }
            imageIndex++;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return element;
    }

    /**
     * 获取下一个label
     * @return
     */
    public double[] getNextLabel() {
        double[] result = new double[10];
        try {
            if (labelIndex >= labelNumber) {
                this.imageBin.close();
                this.labelBin.close();
                return null;
            }
            result[(int) labelBin.read()] = 1;
            labelIndex++;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
    }

    /**
     * 重新打开流
     * @param images
     * @param labels
     */
    public void open(String images, String labels) {
        try {
            this.imageBin = new BufferedInputStream(new FileInputStream(images));
            byte[] bytes = new byte[4];
            this.imageBin.read(bytes, 0, 4);
            // 读取魔数
            if (!"00000803".equals(bytesToHex(bytes))) {
                throw new RuntimeException("Please select the correct file!");
            } else {
                this.imageBin.read(bytes, 0, 4);
                // 读取样本总数
                this.imageNumber = Integer.parseInt(bytesToHex(bytes), 16);
                this.imageBin.read(bytes, 0, 4);
                // 读取每行所含像素点数
                xPixel = Integer.parseInt(bytesToHex(bytes), 16);
                this.imageBin.read(bytes, 0, 4);
                // 读取每列所含像素点数
                yPixel = Integer.parseInt(bytesToHex(bytes), 16);
            }

            this.labelBin = new BufferedInputStream(new FileInputStream(labels));
            bytes = new byte[4];
            this.labelBin.read(bytes, 0, 4);
            if (!"00000801".equals(bytesToHex(bytes))) {
                throw new RuntimeException("Please select the correct file!");
            } else {
                this.labelBin.read(bytes, 0, 4);
                this.labelNumber = Integer.parseInt(bytesToHex(bytes), 16);
            }
            this.labelIndex = 0;
            this.imageIndex = 0;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    /**
     * change bytes into a hex string.
     *
     * @param bytes bytes
     * @return the returned hex string
     */
    public static String bytesToHex(byte[] bytes) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < bytes.length; i++) {
            String hex = Integer.toHexString(bytes[i] & 0xFF);
            if (hex.length() < 2) {
                sb.append(0);
            }
            sb.append(hex);
        }
        return sb.toString();
    }
}
