var React = require('react');

var LineChart = require("react-chartjs").Line;

const styles = {

    main:{
        display: 'flex',
        alignItems: 'center',
        flexDirection: 'column',
    },
    container:{
        width: '100%',
    }
};

var Train = React.createClass({

  getInitialState(){

        return({open: false, trainLossHistory: [], trainAccuracyHistory: [], testLossHistory: [], testAccuracyHistory: []});

  },

  componentDidMount(){

        if (this.state.ws_train) {
            return false;
        }
        let ws_train = new WebSocket("ws://localhost:8888/train");

        ws_train.onopen = this.onTrainOpen;
        ws_train.onclose = this.onTrainClose;
        ws_train.onmessage = this.onTrainMessage;

        ws_train.onerror = function(evt) {
            //print("ERROR: " + evt.data);
        }



        let ws_test = new WebSocket("ws://localhost:8888/test");

        ws_test.onopen = this.onTestOpen;
        ws_test.onclose = this.onTestClose;
        ws_test.onmessage = this.onTestMessage;

        ws_test.onerror = function(evt) {
            //print("ERROR: " + evt.data);
        }

        this.setState({ws_test: ws_test, ws_train: ws_train})




  },

  onTrainMessage(evt){
    let data = JSON.parse(evt.data);
    let epoch = data.Epoch + (data.Batch/data.Batches)

    let tlh = this.state.trainLossHistory;
    tlh.push({x:epoch, y:data.Loss});


    let tah = this.state.trainAccuracyHistory;
    tah.push({x:epoch, y:data.Accuracy});

    this.setState({trainLossHistory: tlh, trainAccuracyHistory: tah});
  },

    onTrainOpen(evt){
    this.setState({open: true});
  },

  onTrainClose(evt){
    this.setState({open: false, ws: null});
  },

    onTestMessage(evt){
    let data = JSON.parse(evt.data);
    let epoch = data.Epoch + 1

    let tlh = this.state.testLossHistory;
    tlh.push({x:epoch, y:data.Loss});


    let tah = this.state.testAccuracyHistory;
    tah.push({x:epoch, y:data.Accuracy});

    this.setState({ testLossHistory: tlh, testAccuracyHistory: tah});
  },

  onTestOpen(evt){
    this.setState({open: true});
  },

  onTestClose(evt){
    this.setState({open: false, ws: null});
  },

  getDataSet(){

  var data = {
    labels:[],
    datasets: [
        {
          label: 'Train Loss',
          backgroundColor : 'rgba(0,0,0,0)',
          borderColor : '#000',
          pointBackgroundColor : '#444',
          pointBorderColor : "#fff",
          data : this.state.trainLossHistory
        },
        {
          label: 'Train Accuracy',
          backgroundColor : 'rgba(0,0,0,0)',
          borderColor : '#500',
          pointBackgroundColor : '#444',
          pointBorderColor : "#fff",
          data : this.state.trainAccuracyHistory
        },

        {
          label: 'Test Loss',
          backgroundColor : 'rgba(0,0,0,0)',
          borderColor : '#050',
          pointBackgroundColor : '#444',
          pointBorderColor : "#fff",
          data : this.state.testLossHistory
        },
        {
          label: 'Test Accuracy',
          backgroundColor : 'rgba(0,0,0,0)',
          borderColor : '#005',
          pointBackgroundColor : '#444',
          pointBorderColor : "#fff",
          data : this.state.testAccuracyHistory
        }
      ]
    };

    return data;

  },

	render: function() {
    if (!this.state.open){
      return(<p>Socket closed</p>)
    }


    var options = {
    responsive: true,
    scales: {
            xAxes: [{
                type: 'linear',
                position: 'bottom'
            }]
        }
    };

		return (
          <div>
            <h2>Loss Graph</h2>
            <LineChart data={this.getDataSet()} options={options}/>
          </div>
		);
	}
});

export default Train
