var webpack = require('webpack');
var path = require('path');

module.exports = {
  context: path.join(__dirname, 'src'),
  entry: ['whatwg-fetch', 'babel-polyfill', './main.jsx'],

  output: {
    filename: 'bundle.js',
    path: path.join(__dirname , 'build'),
    //publicPath: 'http://localhost:8080/build/'
  },

  module: {

    loaders: [
        {
            test: /\.jsx?$/,
            exclude: /(node_modules|bower_components)/,
            loader: 'babel', // 'babel-loader' is also a legal name to reference
            query: {
              presets: ['react', 'es2015']
            }
        },

        {
            test: /\.js?$/,
            exclude: /(node_modules|bower_components)/,
            loader: 'babel', // 'babel-loader' is also a legal name to reference
            query: {
              presets: ['es2015']
            }
        }

        ]
    },
    node: {
      fs: "empty"
    }
};
