#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <limits.h>
#include <map>
#include <time.h>
#include <omp.h>
#include <algorithm>
#define MAX 10000

using namespace std;


/*	cluster_number is a 1D vector that contains the cluster number for each of the data items. 
	data is a 2D vector, where each element is a 1D vector comprising of 4 double values, each value represents the value of a feature for a particular
	Iris flower.
	means is a 2D vector that contains the coordinates of the means of the K clusters 
	cluster_item is a map that maintains a mapping between the cluster number and the list of items that belong to it */


vector<int> cluster_number(10000,-1);
vector<vector<double> > data;
vector<vector<double> > means;
map<int,vector<int> >cluster_item;



double Euclidean_Distance( vector<double>& point1 , vector<double>& point2 )
{
	int i,n = point1.size();
	double distance = 0;

	for ( i = 0 ; i < n ; i++ )
	{
		distance += pow( (point1[i]-point2[i]),2 );
	}
	return sqrt(distance);
}


/* Calculate_Mean() :
   I/P : indices of the data points of a particular cluster.
   O/P : coordinates of the mean of that cluster 
*/


vector<double> Calculate_Mean( vector<int>& indices )
{
	int i,j,n = indices.size();				/* indices is a 1D int vector that contains the indices of the data points for a particular cluster number */
	vector<double> feature_vector;			/* feature_vector is the double vector of size 4, that contains the values of the 4 features for a data point */
	vector<double> row(4,0);				/* row vector is of size 4, since there are only 4 features in the data set */


	for ( i = 0 ; i < n ; i++ )
	{
		feature_vector = data[indices[i]];		/* For every data point belonging to the cluster, fetch the feature vector */

		for ( j = 0 ; j < feature_vector.size() ; j++ )		/* Now add up the values of the jth feature for all the data points */
		{
			row[j] += feature_vector[j];		
		}
	}


	for ( j = 0 ; j < feature_vector.size() ; j++ )	/* Now calculate the position of the mean of the cluster */
	{
		row[j] /= n;
	}

	return row;
}


/* Find_New_Mean():
   I/P:	Does not take any input.
   O/P: Calculates the data points corresponding each cluster and then calculates the coordinates of the mean for each cluster */


void Find_New_Mean()
{
	int i;
	double mean;
	vector<double> row;
	map<int,vector<int> >:: iterator mp;
	vector<int> :: iterator vec;

	/* Calculating the indices of the data points for each cluster number */

	for ( i = 0 ; i < cluster_number.size() ; i++ )
	{
		cluster_item[cluster_number[i]].push_back(i);
	}

	for ( mp = cluster_item.begin() ; mp != cluster_item.end() ; mp++ )
	{
		row = Calculate_Mean( (mp->second) );
		means[mp->first] = row;					/* mp->first means the cluster number. So this is coordinates of the mean for the cluster number given by mp->first  */
	}
}


void K_Means_Parallel()
{

	int i,j,iter,minimum,index,chunk_size;
	double distance;
	bool noChange;


	/* The size of the data set is 10000 */

	 omp_set_num_threads(100);			/* 100 threads have been launched */
	 chunk_size = data.size()/100;		/* chunk_size is the size of the data chunk to be dealt by each thread */


	for ( iter = 1 ; iter <= MAX ; iter++ )
	{
		noChange = true;

		cout<<iter<<endl;

		#pragma omp parallel for private(j,minimum,distance,index) schedule( dynamic , chunk_size )


		/* Intuition behind parallelization:

			For every data point in the data set, the distance of that data point from each of the mean points needs to be calculated.
			The calculation of these distances for one data point is completely independent of the other data points in the data set.
			So, the iterations of the below for loop can be parallelized.


			100 threads have been launched each managing 100 iterations of the below for loop.
			Theoretically, one thread can be launched for each iteration i.e. 10000 threads.
			However, this will increase the work of the thread scheduler, and there will be more of context switching.

			
			Variables j,minimum,distance,index have been made private for each thread that have been launched.
			Keeping them shared for each thread, will lead to data race conditions, and subsequently wrong results

		*/


		for ( i = 0 ; i < data.size() ; i++ )
		{
			minimum = INT_MAX;

			for ( j = 0 ; j < means.size() ; j++ )
			{
				distance = Euclidean_Distance( data[i] , means[j] );

				if ( minimum > distance )
				{
					minimum = distance;
					index   = j;
				}
			}

			if ( index != cluster_number[i] )
			{
				noChange = false;
			}

			cluster_number[i] = index;
		}
		

		if( noChange )
		{
			break;
		}

		Find_New_Mean();

		/* Once the new mean has been found, the map cluster_item is cleared */

		for ( int m = 0 ; m < means.size() ; m++ )
		{
			cluster_item[m].clear();
		}

	}
}


/* Tokenize() takes a string as input and spilts into 4 double values using comma as a delimiter */ 

vector<double> Tokenize( string line )
{
	int i;
	string temp;
	vector<double> row;

	for( i = 0 ; i < line.size() ; i++ )
	{
		while( line[i] != ',' )
		{
			temp += line[i];				/* Keep on appending the numerical contents of line to temp, unless a comma appears in the string */
			i++;
		}

		row.push_back(stod(temp));			/* Once a comma appears, push the double equivalent of the string to the vector row */
		temp.clear();
	}
	return row;	
}



/* parseCSV2double() takes the name of a csv file as input and returns a 2D vector, where each element is a 1D double vector comprising the values of the
   4 features */

vector<vector<double> > parseCSV2double ( char *filename )
{
	int n,l = 0,first_comma,last_comma;
	string line;
	vector<double> row;
	vector<vector<double> > res;
	ifstream InputFile( filename );

	while( getline( InputFile , line )  )
	{
		if(l)
		{
			n = line.size();
			first_comma = line.find_first_of(',');  						/* Find the first occurence of comma in the string line*/
			last_comma  = line.find_last_of(',');							/* Find the  last occurence of comma in the string line */
			line = line.substr(first_comma+1,last_comma-first_comma);		/* Pick up all the content between first and last comma */

			/* Now line contains only the value of the 4 features separated by comma */ 
			row = Tokenize(line);
			res.push_back(row);
		}
		l++;
	}
	return res;
}



int main( int argc , char *argv[] )
{
	int i,K,size;
	double begin_time,end_time,diff;

	data = parseCSV2double(argv[1]);

	cout<<"Enter the number of clusters:\t";
	cin>>K;

	/* Choosing K random points for mean initialization */
	 
	srand(time(0));

	for( i = 1 ; i <= K ; i++ )
	{
		//means.push_back(data[rand()%data.size()]);
		means.push_back(data[i*100]);
	}

	begin_time = omp_get_wtime();

	K_Means_Parallel();

	end_time = omp_get_wtime();
	diff = end_time - begin_time;

	cout<<"\nTime required for "<<K<<" clusters is "<<diff<<" seconds "<<endl;

}	
