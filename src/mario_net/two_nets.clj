(ns mario-net.two-nets
  (:require [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [mikera.image.core :as i]
            [cortex.description :as desc]
            [cortex.network :as net]))

(defn index->label
  [idx c]
  (assoc (vec (repeat c 0))
         idx 1))

(defn build-dataset
  [x-offset y-offset]
  (let [d (->> (io/resource "data/16x16-map.edn")
               (slurp)
               (read-string))
        sprite-count (inc (apply max (map second d)))
        max-y (apply max (map second (keys d)))]
    (println "Sprite count:" sprite-count)
    (println "Max-y:" max-y)
    (->> d
         (map (fn [[[x y] i]]
                (if-let [o (get d [(+ x x-offset) (+ y y-offset)])]
                  [[(double (/ i (dec sprite-count))) (double (/ y max-y))] (index->label i sprite-count)])))
         (filter identity))))

(defn load-index
  []
  (->> (io/resource "data/16x16-index.edn")
       (slurp)
       (read-string)))

(def network-desc
  [(desc/input 2)
   (desc/linear->relu 50)
   (desc/softmax 33)])

(defn train-network
  [x-offset y-offset]
  (let [dataset (build-dataset x-offset y-offset)
        training-data (mapv first dataset)
        training-labels (mapv second dataset)
        network (desc/build-and-create-network network-desc)]
    (println "Training network:" (count training-data) "inputs...")
    (net/train-until-error-stabilizes network training-data training-labels)))

(defn sample-index-from
  [v]
  (let [r (* 2 (rand))]
    (loop [i 0]
      (let [sum (reduce + (take (inc i) v))]
        (if (> sum r)
          i
          (recur (inc i)))))))

(defn predict
  [x-net y-net left-index above-index y]
  (sample-index-from (seq (m/add (first (net/run x-net [[left-index y]]))
                                 (first (net/run y-net [[above-index y]]))))))

(defn nets->picture
  [x-net y-net]
  (->> (for [x (range 15)
             y (range 15)]
         [x y])
       (reduce (fn [eax [x y]]
                 (let [prediction (predict x-net y-net
                                           (get eax (dec x) 13)
                                           (get eax (dec y) 13)
                                           y)]
                   (assoc eax [x y] prediction)))
               {})))


(defn picture->image
  [pic]
  (let [[w h] [15 15]
        index (read-string (slurp "resources/data/16x16-index.edn"))
        img (i/new-image (* 16 w) (* 16 h))]
    (doseq [x (range w)
            y (range h)]
      (let [sub-image (i/sub-image img (* 16 x) (* 16 y) 16 16)]
        (i/set-pixels sub-image (int-array (get index (get pic [x y]))))))
    img))
