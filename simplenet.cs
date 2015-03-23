using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.IO;
using System.Xml;


namespace NeuralNet
{
	
	
	public class SimplePerceptron
	{
		
		public OrderedDictionary inputs, outputs;
		protected string topofile;
		
		public double fitness = 0;
		public int roundswon = 0, roundsplayed = 0;
			
		protected List<double[]> neurons;
		protected List<double[]> bias;
		protected List<double[,]> links;
		
		protected Random rnd;
		protected int[] structure;
		
		protected double maxlink = 1.0;
		protected double maxbias = 10.0;
		
		/*
		public SimplePerceptron (int inputs, int outputs, int[] hidden)
		{
			structure = hidden;
			rnd = new Random();
			
			#region "make neurons"
			neurons = new List<double[]>(hidden.Length+2); //one for each hidden + in + out
			bias = new List<double[]>(hidden.Length+1); //one for each hidden + out
			
			neurons.Add(new double[inputs]);
			
			for (int i = 0; i < hidden.Length; i++) {
				neurons.Add(new double[hidden[i]]);
				bias.Add(new double[hidden[i]]);
			}
			
			neurons.Add(new double[outputs]);
			bias.Add(new double[outputs]);
			#endregion
			#region "make connections"
			links = new List<double[,]>(hidden.Length+1);
			links.Add(new double[inputs,hidden[0]]); //connection input->hidden0
			
			for (int i = 1; i < hidden.Length; i++) {
				links.Add(new double[hidden[i-1],hidden[i]]); //connection hiddeni->hiddeni-1
			}
			links.Add(new double[hidden[hidden.Length-1],outputs]); //connection hiddenlast->output
			
			#endregion
			
			RandomizeNeurons();
			RandomizeLinks();
			
		}*/
		public SimplePerceptron (string xmltopology)
		{
			rnd = new Random();
			topofile = xmltopology;
			XmlDocument xml = new XmlDocument();
			xml.Load(xmltopology);
			
			XmlNode main = xml.SelectSingleNode("network");
			
			XmlNode mainins = main.SelectSingleNode("inputs");
			XmlNodeList ins = mainins.SelectNodes("input");
			inputs = new  OrderedDictionary();
			foreach(XmlNode node in ins) {
				inputs.Add(node.Attributes["name"].Value,0);
			}

			XmlNode mainouts = main.SelectSingleNode("outputs");
			XmlNodeList outs = mainouts.SelectNodes("output");
			outputs = new OrderedDictionary();
			foreach(XmlNode node in outs) {
				outputs.Add(node.Attributes["name"].Value,0);
			}
			
			#region "make layers"
			XmlNode mainlayers = main.SelectSingleNode("layers");
			XmlNodeList layers = mainlayers.SelectNodes("layer");
			structure = new int[layers.Count];
			
			bias = new List<double[]>(layers.Count + 1); //bias for layers + outputs
			neurons = new List<double[]>(layers.Count + 2); //one for each hidden + in + out
			neurons.Add(new double[inputs.Count]); //add the input layer
			
			int k=0;
			foreach(XmlNode node in layers) { //add the hidden layers
				structure[k] = int.Parse(node.Attributes["size"].Value);
				neurons.Add(new double[structure[k]]);
				bias.Add(new double[structure[k]]);
				k++;
			}
			
			neurons.Add(new double[outputs.Count]); //add the output layer
			bias.Add(new double[outputs.Count]); //add biases for output layer
			
			#endregion
			
			#region "make links"
			
			links = new List<double[,]>(layers.Count+1);
			links.Add(new double[inputs.Count,bias[0].Length]); //connection input->hidden0
			
			for (int i = 1; i < bias.Count; i++) {
				links.Add(new double[bias[i-1].Length,bias[i].Length]); //connection hiddeni->hiddeni-1
			}
			
			#endregion
			
			
			
			RandomizeNeurons();
			RandomizeLinks();
			
			
		}
		
		public void RandomizeNeurons()
		{
			
			foreach(double[] d in bias)
				for (int i = 0; i < d.Length; i++) {
					d[i] = 2*maxbias*(rnd.NextDouble()-0.5);
				}
			
		}
		public void RandomizeLinks()
		{
			
			foreach(double[,] matrix in links)
			{
				for (int i = 0; i < matrix.GetLength(0); i++) {
					for (int j = 0; j < matrix.GetLength(1); j++) {
						matrix[i,j] = 2*maxlink*(rnd.NextDouble()-0.5);
					}
				}
				
			}
			
		}
		
		public void ClearInputs() {
		
			for(int i=0;i<inputs.Count;i++)
				inputs[i] = 0;
			
		}
		
		public void Update()
		{
			#region "copy the input from dictionary"
			double[] vals = new double[inputs.Count];
			inputs.Values.CopyTo(vals,0);
			
			for (int i = 0; i < inputs.Count; i++) {
				neurons[0][i] = vals[i];
			}
			#endregion
			
			for (int layer = 1; layer < neurons.Count; layer++) { //loop over the layers+out
				
				//loop over the neurons to be updated
				for (int output = 0; output < neurons[layer].Length; output++) { 
					neurons[layer][output] = bias[layer-1][output];
					
					//loop over the neurons in the previous layer
					for (int input = 0; input < neurons[layer-1].Length; input++) {
						neurons[layer][output] += neurons[layer-1][input]*links[layer-1][input,output];
					}
					
					neurons[layer][output] = Math.Tanh(neurons[layer][output]);
					
				}
			}
			
			#region "copy the output to dictionary"
			for (int i = 0; i < outputs.Count; i++) {
				outputs[i] = neurons[neurons.Count-1][i];
			}
			#endregion
			
			
		}
		
		
		public static SimplePerceptron Mix(SimplePerceptron mom, SimplePerceptron dad, double mutationrate)
		{
			Random rnd = mom.rnd; //mom random seed rulez!
			
			SimplePerceptron son = new SimplePerceptron(mom.topofile);
			
			son.maxbias = mom.maxbias;
			son.maxlink = mom.maxlink;
			
			//for each non input layer... copy the biases
			for (int layer = 0; layer < mom.bias.Count; layer++) {
				
				int momlim = rnd.Next(0,mom.bias[layer].Length);
				for (int i = 0; i < mom.bias[layer].Length; i++) {
					son.bias[layer][i] = (i<=momlim)? mom.bias[layer][i] : dad.bias[layer][i];
					if(dad.rnd.NextDouble() < mutationrate) //mutate!
						son.bias[layer][i] = 2*son.maxbias*(rnd.NextDouble()-0.5);
					
				}
				//copy connections
				for (int x = 0; x < son.links[layer].GetLength(0); x++) {
					momlim = rnd.Next(0,son.links[layer].GetLength(1));
					for (int y = 0; y < son.links[layer].GetLength(1); y++) {
						son.links[layer][x,y] = (y<=momlim)? mom.links[layer][x,y] : dad.links[layer][x,y];
						if(dad.rnd.NextDouble() < mutationrate) //mutate!
							son.links[layer][x,y] = 2*son.maxlink*(rnd.NextDouble()-0.5);
					}
					
				}
				
			}
			
			return son;
		}
		
		
		public void Dump(StreamWriter w)
		{
			
			w.WriteLine("fitness {0}",fitness);
			#region "write all biases for each layer"
			w.WriteLine("biaslayers {0}",bias.Count);
			for (int layer = 0; layer < bias.Count; layer++) {
				for (int i = 0; i < bias[layer].Length; i++) {
					w.Write("{0} ",bias[layer][i]);
				}
				w.WriteLine();
			}
			#endregion
			#region "write all links for each layer"
			w.WriteLine("linklayers {0}",links.Count);
			for (int layer = 0; layer < links.Count; layer++) {
				w.WriteLine("linksize {0} {1}",links[layer].GetLength(0),links[layer].GetLength(1));
				for (int i = 0; i < links[layer].GetLength(0); i++) {
					for (int j = 0; j < links[layer].GetLength(1); j++) {
						w.Write("{0} ",links[layer][i,j]);
					}
					w.WriteLine();
				}
			}
			#endregion
			
			
		}

		public void DumpXML (string filename)
		{
			XmlDocument doc = new XmlDocument ();
			XmlNode node = doc.CreateXmlDeclaration ("1.0", "UTF-8", null);
			doc.AppendChild (node);

			XmlNode root = doc.CreateElement ("network");
			XmlAttribute attr = doc.CreateAttribute ("inputs");
			attr.Value = neurons [0].Length.ToString ();
			root.Attributes.Append (attr);

			attr = doc.CreateAttribute ("outputs");
			attr.Value = neurons [neurons.Count - 1].Length.ToString ();
			root.Attributes.Append (attr);


			for (int i = 0; i < bias.Count; i++) {
				XmlNode layer = doc.CreateElement ("layer");
				attr = doc.CreateAttribute("size");
				attr.Value = bias[i].Length.ToString();
				layer.Attributes.Append(attr);
				layer.InnerText = " ";
				for (int j = 0; j < bias[i].Length; j++) {
					layer.InnerText += bias[i][j].ToString()+" ";
				}
				root.AppendChild(layer);
			}

			for (int k = 0; k < links.Count; k++) {

				XmlNode lnk = doc.CreateElement ("links");
				attr = doc.CreateAttribute("nx");
				attr.Value = links[k].GetLength(0).ToString();
				lnk.Attributes.Append(attr);
				attr = doc.CreateAttribute("ny");
				attr.Value = links[k].GetLength(1).ToString();
				lnk.Attributes.Append(attr);
				lnk.InnerText= " ";
				for (int i = 0; i < links[k].GetLength(0); i++) {
					for (int j = 0; j < links[k].GetLength(1); j++) {
						lnk.InnerText += links[k][i,j].ToString()+" ";
					}
					lnk.InnerText += "\n";
				}
				root.AppendChild(lnk);
			}



			doc.AppendChild(root);

			doc.Save(filename);
		}

		public void DumpXML (XmlDocument doc)
		{

			XmlNode main = doc.SelectSingleNode("networks");

			XmlNode root = doc.CreateElement ("network");


			#region "inputs"

			XmlNode xinputs = doc.CreateElement("inputs");
			string[] keys = new string[inputs.Count];
			inputs.Keys.CopyTo(keys,0);

			for (int i = 0; i < inputs.Count; i++) {
				XmlNode n = doc.CreateElement("input");
				XmlAttribute a = doc.CreateAttribute("name");
				a.Value = keys[i];
				n.Attributes.Append(a);
				xinputs.AppendChild(n);
			}

			root.AppendChild(xinputs);

			#endregion
			#region "outputs"

			XmlNode xoutputs = doc.CreateElement("outputs");
			keys = new string[outputs.Count];
			outputs.Keys.CopyTo(keys,0);
			
			for (int i = 0; i < outputs.Count; i++) {
				XmlNode n = doc.CreateElement("output");
				XmlAttribute a = doc.CreateAttribute("name");
				a.Value = keys[i];
				n.Attributes.Append(a);
				xoutputs.AppendChild(n);
			}

			root.AppendChild(xoutputs);
			
            #endregion
			#region "layers"

			XmlNode xlayers = doc.CreateElement("layers");

			for (int i = 0; i < bias.Count; i++) {
				XmlNode n = doc.CreateElement("layer");
				XmlAttribute a = doc.CreateAttribute("size");
				a.Value = bias[i].Length.ToString();
				n.Attributes.Append(a);

				string biasstring = "";
				for (int j = 0; j < bias[i].Length; j++) {
					biasstring += bias[i][j]+" ";
				}
				n.InnerText = biasstring;
				xlayers.AppendChild(n);
			}
			
			root.AppendChild(xlayers);

			#endregion 
			#region "links"
			
			XmlNode xlinks = doc.CreateElement("links");

			for (int k = 0; k < links.Count; k++) {
				XmlNode n = doc.CreateElement("link");
				//XmlAttribute a = doc.CreateAttribute("x");
				//a.Value = bias[i].Length.ToString();
				//n.Attributes.Append(a);


				for (int i = 0; i < links[k].GetLength(0); i++) {
					for (int j = 0; j < links[k].GetLength(1); j++) {
						n.InnerText += links[k][i,j].ToString()+" ";
					}
					n.InnerText += "\n";
				}

				xlinks.AppendChild(n);
			}
			
			root.AppendChild(xlinks);
			
			#endregion 

			main.AppendChild(root);

		}




		public void Load (string filename)
		{
			StreamReader r = new StreamReader (filename);
			char[] delims = new char[] {' ','\t'};
			
			r.ReadLine (); //fitness
			
			r.ReadLine (); //biascount
			for (int layer = 0; layer < bias.Count; layer++) {
				string line = r.ReadLine ();
				string[] words = line.Split (delims, StringSplitOptions.RemoveEmptyEntries);
				for (int i = 0; i < bias[layer].Length; i++) {
					bias [layer] [i] = double.Parse (words [i]);
				}
			}
			
			r.ReadLine (); //linkslayers
			for (int layer = 0; layer < links.Count; layer++) {
				r.ReadLine (); //linksize
				for (int i = 0; i < links[layer].GetLength(0); i++) {
					string line = r.ReadLine ();
					string[] words = line.Split (delims, StringSplitOptions.RemoveEmptyEntries);
					for (int j = 0; j < links[layer].GetLength(1); j++) {
						links [layer] [i, j] = double.Parse (words [j]);
					}
				}
			}
			
			Console.Write("selfcheck: ");
			Update ();
			for (int i=0; i<neurons[neurons.Count-1].Length; i++) {
				Console.Write("{0} ",neurons[neurons.Count-1][i]);
			}
			Console.WriteLine("");

			r.Close();r.Dispose();
		}
		
		public void Load(StreamReader restartfile)
		{
		
			char[] delims = new char[] {' ','\t'};
			
			restartfile.ReadLine(); //fitness
			
			restartfile.ReadLine(); //biascount
			for (int layer = 0; layer < bias.Count; layer++) {
				string line = restartfile.ReadLine();
				string[] words = line.Split(delims,StringSplitOptions.RemoveEmptyEntries);
				for (int i = 0; i < bias[layer].Length; i++) {
					bias[layer][i] = double.Parse(words[i]);
				}
			}
			
			restartfile.ReadLine(); //linkslayers
			for (int layer = 0; layer < links.Count; layer++) {
				restartfile.ReadLine(); //linksize
				for (int i = 0; i < links[layer].GetLength(0); i++) {
					string line = restartfile.ReadLine();
					string[] words = line.Split(delims,StringSplitOptions.RemoveEmptyEntries);
					for (int j = 0; j < links[layer].GetLength(1); j++) {
						links[layer][i,j] = double.Parse(words[j]);
					}
				}
			}
			
			
		}
		
		public static SimplePerceptron LoadXML (string filename)
		{

			XmlDocument doc = new XmlDocument ();
			doc.Load (filename);

			XmlNode root = doc.SelectSingleNode ("network");
			int ninputs = int.Parse (root.Attributes ["inputs"].Value);
			int noutputs = int.Parse (root.Attributes ["outputs"].Value);

			XmlNodeList layers = root.SelectNodes ("layer");
			int[] hidden = new int[layers.Count - 1];
			for (int i = 0; i < layers.Count-1; i++) {
				hidden [i] = int.Parse (layers [i].Attributes ["size"].Value);
			}

			SimplePerceptron result = new SimplePerceptron ("ai4topology.xml");

			for (int i = 0; i < layers.Count; i++) {

				int x = int.Parse (layers [i].Attributes ["size"].Value);
				//int y = int.Parse(layers[i].Attributes["ny"].Value);

				string[] words = layers [i].InnerText.Split (new char[] {' ','\n'}, StringSplitOptions.RemoveEmptyEntries);
				for (int j = 0; j < x; j++) {
					//Console.WriteLine(words[j]);
					result.bias [i] [j] = double.Parse (words [j]);
				}

			}

			layers = root.SelectNodes ("links");
			for (int k = 0; k < layers.Count; k++) {

				int x = int.Parse (layers [k].Attributes ["nx"].Value);
				int y = int.Parse(layers[k].Attributes["ny"].Value);
				string[] words = layers [k].InnerText.Split (new char[] {' ','\n'}, StringSplitOptions.RemoveEmptyEntries);

				for (int i = 0; i < x; i++) {
					for (int j = 0; j < y; j++) {
						result.links[k][i,j] = double.Parse(words[i*y+j]);
					}
				}


			}


			return result;

		}

	}
	
	
	
}

