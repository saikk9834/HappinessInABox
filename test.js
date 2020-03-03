const express = require('express');
const app = express();
const fs = require('fs');
const HTMLParser = require('node-html-parser');
const abc = require('./abc');

require('dotenv').config();
app.use(express.json());
app.get('/getvalues/', (req,res)=>{
    fs.readFile('build.html', "utf-8", (err, data) => {
        if (err) throw err;
        t = data.replace(/\s\s/g,'').split('\n')[0];
        var arr=[];
        var root = HTMLParser.parse(t);
        root.structuredText.split('\n').forEach(element => {
            console.log(element.replace(/\s/g,'').split(':'));
            var val = element.replace(/\s/g,'').split(':')[1];
            arr.push(val);
        });
        res.send(arr);
      });
});

app.post('/clearCache/', (req,res) =>{
    var app_name = req.body.app_name;
    var env = req.body.env;
    var userid = req.body.userid;
    var region = req.body.region;
    var login_type = req.body.login_type;
    var response = abc.clear(app_name,env,userid,region,login_type);
    res.send(response);
  });

app.listen(3000, () => console.log("App listening on port 3000!!"));