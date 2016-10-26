(ns mario-net.pixel-try-two
  (:require [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [mikera.image.core :as i]
            [mikera.image.colours :as colors]
            [taoensso.nippy :as nippy]
            [cortex.nn.network :as net]
            [cortex.nn.description :as desc]))

;; Sky is -10709764, index 0.

(def index
  (-> (io/resource "data/index.edn")
      (slurp)
      (read-string)))

(def reverse-index
  (into {} (map (fn [[k v]] [v k]) index)))

(defn- color->one-hot
  [c]
  (let [dim (count index)
        idx (get index c)]
    (assoc (vec (repeat dim 0))
           idx 1)))

(defn- img->input
  [img x y w h]
  (flatten [(->> (i/sub-image img (- x 4) (- y 3) 4 8)
                               (i/get-pixels)
                               (map color->one-hot))
                          (->> (i/sub-image img x (inc y) 1 4)
                               (i/get-pixels)
                               (map color->one-hot))
                          [(double (/ x w))
                           (double (/ y h))]]))

(defn- build-dataset
  []
  (let [img (i/load-image-resource "1-1.png")
        w (i/width img)
        h (i/height img)]
    (for [x (range 4 w)
          y (range 3 (- h 4))]
      [(m/array :vectorz (img->input img x y w h))
       (-> img
           (i/sub-image x y 1 1)
           (i/get-pixels)
           (first)
           (color->one-hot))])))

(def network-desc
  [(desc/input 326)
   (desc/linear->relu 400)
   (desc/linear->relu 400)
   (desc/linear->softmax (count index))])

(defn train-network
  []
  (let [dataset (build-dataset)
        training-data (mapv first dataset)
        training-labels (mapv second dataset)
        network (desc/build-and-create-network network-desc)]
    (println "Training network:" (count training-data) "inputs...")
    (net/train-n-epochs network 10 training-data training-labels :batch-size 100)))

(defn net->file
  [net path]
  (with-open [w (clojure.java.io/output-stream path)]
    (->> net
         desc/network->description
         nippy/freeze
         (.write w))))

(defn sample-index-from
  [v]
  (let [m (reduce + v)
        r (* m (rand))]
    (loop [i 0]
      (let [sum (reduce + (take (inc i) v))]
        (if (> sum r)
          i
          (recur (inc i)))))))

(defn net->img
  [net w h]
  (let [img (i/new-image w h)]
    (i/fill-rect! img 0 0 w h (colors/color (get reverse-index 0)))
    (doseq [x (range 4 w)
            y (reverse (range 3 (- h 4)))]
      (if (= y 100)
        (println x))
      (let [c (->> (net/run net [(m/array :vectorz (img->input img x y w h))])
                   (first)
                   (vec)
                   (sample-index-from)
                   (reverse-index))]
        (i/set-pixel img x y c)))
    img))
