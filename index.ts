import fs from "fs";
import * as tf from "@tensorflow/tfjs-node";
import * as mobilenet from "@tensorflow-models/mobilenet";

const assets = `${__dirname}/assets`;

const classify = async (decodedImage: any) => {
    let model = await mobilenet.load();
    let predictions = await model.classify(decodedImage);
    console.log(`Predictions: ${JSON.stringify(predictions, undefined, 2)}`);
}

const App = () => {
    fs.readdir(assets, (err, files) => {
        files.forEach(async (file) => {
            let fileLocation = `${assets}/${file}`;
            let image = fs.readFileSync(fileLocation);
            let decodedImage = tf.node.decodeImage(image, 3);

            await classify(decodedImage);
        });
    });
}

App();